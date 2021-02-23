from collections import defaultdict
from functools import partial
import logging
import sys

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import numpy as np
import pandas as pd

from utils import general_utils

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
tqdm().pandas()


def split_sequences(group_data, n_steps_in, n_steps_out, x_cols, y_col, additional_columns, step=1, lag_fns=[]):
    X, y = list(), list()
    additional_col_map = defaultdict(list)
    group_data = group_data.sort_values('Date')
    for i, lag_fn in enumerate(lag_fns):
        group_data[f'lag_{i}'] = lag_fn(group_data[y_col])
    steps = list(range(0, len(group_data), step))
    if step != 1 and steps[-1] != (len(group_data) - 1):
        steps.append((len(group_data) - 1))
    for i in steps:
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(group_data):
            break
        # gather input and output parts of the pattern
        if len(x_cols) == 1:
            x_cols = x_cols[0]
        seq_x, seq_y = group_data.iloc[i:end_ix, :][x_cols].values, group_data.iloc[end_ix:out_end_ix, :][[y_col] + [f'lag_{i}' for i in range(len(lag_fns))]].values
        for col in additional_columns:
            additional_col_map[col].append(group_data.iloc[end_ix][col])
        X.append(seq_x)
        y.append(seq_y)
    additional_column_items = sorted(additional_col_map.items(), key=lambda x: x[0])
    return (np.array(X), np.array(y), *[i[1] for i in additional_column_items])


def _apply_df(args):
    df, func, key_column = args
    result = df.groupby(key_column).progress_apply(func)
    return result


def almost_equal_split(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out


def mp_apply(df, func, key_column):
    workers = 6
    # pool = mp.Pool(processes=workers)
    key_splits = almost_equal_split(df[key_column].unique(), workers)
    split_dfs = [df[df[key_column].isin(key_list)] for key_list in key_splits]
    result = process_map(_apply_df, [(d, func, key_column) for d in split_dfs], max_workers=workers)
    return pd.concat(result)


def sequence_builder(data, n_steps_in, n_steps_out, key_column, x_cols, y_col, additional_columns, lag_fns=[], step=1):
    sequence_fn = partial(
        split_sequences,
        n_steps_in=n_steps_in,
        n_steps_out=n_steps_out,
        x_cols=x_cols,
        y_col=y_col,
        additional_columns=list(set([key_column] + additional_columns)),
        lag_fns=lag_fns,
        step=step
    )

    logging.info('Start building sequences')
    sequence_data = mp_apply(
        data[list(set([key_column] + x_cols + [y_col] + additional_columns))],
        sequence_fn,
        key_column
    )

    logging.info('Prepare dataframe')
    sequence_data = pd.DataFrame(sequence_data, columns=['result'])
    s = sequence_data.apply(lambda x: pd.Series(zip(*[col for col in x['result']])), axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'result'
    sequence_data = sequence_data.drop('result', axis=1).join(s)
    sequence_data['result'] = pd.Series(sequence_data['result'])
    sequence_data[['x_sequence', 'y_sequence'] + sorted(set([key_column] + additional_columns))] = pd.DataFrame(sequence_data.result.values.tolist(), index=sequence_data.index)
    sequence_data.drop('result', axis=1, inplace=True)
    if key_column in sequence_data.columns:
        sequence_data.drop(key_column, axis=1, inplace=True)
    sequence_data = sequence_data.reset_index()
    print(sequence_data.shape)
    sequence_data = sequence_data[~sequence_data['x_sequence'].isnull()]
    return sequence_data


def last_year_lag(col): return (col.shift(364) * 0.25) + (col.shift(365) * 0.5) + (col.shift(366) * 0.25)


if __name__ == '__main__':
    input_data_filename = sys.argv[1]
    output_data_filename = sys.argv[2]
    n_steps_in = int(sys.argv[3])
    logging.info(input_data_filename)
    x_cols = ['Sales', 'Customers', 'Open', 'Promo', 'StateHoliday',
              'SchoolHoliday', 'DayOfWeek_sin', 'DayOfWeek_cos', 'Month_sin',
              'Month_cos', 'Day_sin', 'Day_cos']
    additional_columns = ['Date', 'StoreType', 'Assortment',
                          'CompetitionDistance', 'Promo2', 'PromoInterval',
                          'CompetitionOpenSinceMonth_sin', 'CompetitionOpenSinceMonth_cos',
                          'Promo2SinceWeek_sin', 'Promo2SinceWeek_cos', 'Sales_mean',
                          'Customers_mean', 'Assortment_mean', 'CompetitionDistance_mean']
    logging.info(f'Time-dependant features: {x_cols}')
    logging.info(f'Time-independent features: {additional_columns}')
    logging.info(f'Target Feature: Sales')

    logging.info(f'Load pickle file: {input_data_filename}')
    data = general_utils.open_pickle_file(input_data_filename)

    data = sequence_builder(data=data, n_steps_in=n_steps_in, n_steps_out=47, key_column='Store',
                            x_cols=x_cols, y_col='Sales', additional_columns=additional_columns, lag_fns=[])

    logging.info(f'Save data to {output_data_filename}')
    general_utils.save_pickle_file(output_data_filename, data)