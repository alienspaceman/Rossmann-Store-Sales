from collections import defaultdict
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from . import general_utils

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
tqdm().pandas()

"""
Functions for transformation of cyclical features
"""


def sin_transform(values, n_values):
    return np.sin(2*np.pi*values/n_values)


def cos_transform(values, n_values):
    return np.cos(2*np.pi*values/n_values)


def transform_cyclical_feature(values, cycle_type):
    cycles = {'days_in_week': 7,
              'days_in_month': 31,
              'weeks_in_year': 52,
              'months_in_year': 12
              }
    return sin_transform(values, cycles[cycle_type]), cos_transform(values, cycles[cycle_type])


"""
Functions for data cleaning
"""


def get_closed_dates_info(df):
    idx_to_remove = list(df[(df['Open'] != 1) & (df['Sales'] == 0)].index)
    promo_map = df.iloc[idx_to_remove].groupby('Date')['Promo'].unique().apply(lambda x: x[0])
    state_holiday_map = df.iloc[idx_to_remove].groupby('Date')['StateHoliday'].unique().apply(lambda x: x[0])
    school_holiday_map = df.iloc[idx_to_remove].groupby('Date')['SchoolHoliday'].unique().apply(lambda x: x[0])
    return idx_to_remove, {'Promo': promo_map, 'StateHoliday': state_holiday_map, 'SchoolHoliday': school_holiday_map}


def remove_closed_dates_records(df, idx_to_remove):
    return df.drop(idx_to_remove)


def interpolate_sales_customers(df, method='nearest'):
    logging.info(f'Shape before removal: {df.shape}')

    idx_to_remove, closed_dates_info = get_closed_dates_info(df)
    df_open = remove_closed_dates_records(df, idx_to_remove)
    logging.info(f'Shape after removal: {df_open.shape}')

    df_open = df_open.sort_values(['Store', 'Date']).reset_index(drop=True)
    df_open = df_open.set_index('Date').groupby('Store') \
        .apply(lambda x: x.drop('Store', axis=1).asfreq('D')).reset_index()
    logging.info(f'Shape after filling date gaps: {df_open.shape}')

    df_open['Sales'] = df_open.groupby('Store') \
        .apply(lambda x: x['Sales'].interpolate(method=method)).reset_index(drop=True)
    df_open['Customers'] = df_open.groupby('Store') \
        .apply(lambda x: x['Customers'].interpolate(method=method)).reset_index(drop=True)

    df_open = df_open.set_index('Date')
    for col in list(closed_dates_info.keys()):
        df_open[col].fillna(closed_dates_info[col], inplace=True)
        df_open[col] = df_open[col].interpolate(method=method)
    df_open['StateHoliday'].fillna('0', inplace=True)

    df_open.reset_index(inplace=True)

    df_open['Open'].fillna(0, inplace=True)
    df_open['DayOfWeek'].fillna(df_open['Date'].dt.weekday, inplace=True)
    logging.info(f'Number of missing values in df: {df.isna().sum().sum()}')
    return df_open


def fill_nans_store_df(df):
    competition_cols = ['CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear']
    logging.info(f'Fill NaN in {", ".join(competition_cols)} with 0')
    df[competition_cols] = df[competition_cols].fillna(0)

    promo2_cols =['Promo2SinceWeek', 'Promo2SinceYear']
    logging.info(f'Fill NaN in {", ".join(promo2_cols)} with 0')
    df[promo2_cols] = df[promo2_cols].fillna(0)

    logging.info(f'Fill NaN in PromoInterval with ""')
    df['PromoInterval'] = df['PromoInterval'].fillna('')

    return df


def encode_cat_cols(df, cat_cols):
    logging.info(f'Encode labels in {cat_cols}')
    le = {col: dict(zip(sorted(df[col].unique()), range(len(df[col].unique())))) for col in cat_cols}
    for col in cat_cols:
        df[col] = df[col].map(le[col])
    return df, le


def transform_store_df(df, le_file):
    cat_cols = ['StoreType', 'Assortment', 'PromoInterval']
    df, le = encode_cat_cols(df, cat_cols)

    logging.info('Transform log1p CompetitionDistance')
    df['CompetitionDistance'] = np.log1p(df['CompetitionDistance'])

    logging.info('Transform date features')
    df['CompetitionOpenSinceMonth_sin'], df['CompetitionOpenSinceMonth_cos'] = transform_cyclical_feature(df['CompetitionOpenSinceMonth'], 'months_in_year')
    df.drop('CompetitionOpenSinceMonth', axis=1, inplace=True)
    df['Promo2SinceWeek_sin'], df['Promo2SinceWeek_cos'] = transform_cyclical_feature(df['Promo2SinceWeek'], 'weeks_in_year')
    df.drop('Promo2SinceWeek', axis=1, inplace=True)

    logging.info(f'Save label encoder to {le_file}.pkl')
    general_utils.save_pickle_file(le_file, le)
    return df, le


def transform_sales_df(df, le_file):
    logging.info('Add Day, Month, Year features')
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    logging.info('Transform date features')
    df['DayOfWeek_sin'], df['DayOfWeek_cos'] = transform_cyclical_feature(df['DayOfWeek'], 'days_in_week')
    df.drop('DayOfWeek', axis=1, inplace=True)

    df['Month_sin'], df['Month_cos'] = transform_cyclical_feature(df['Month'], 'months_in_year')
    df.drop('Month', axis=1, inplace=True)

    df['Day_sin'], df['Day_cos'] = transform_cyclical_feature(df['Day'], 'days_in_month')
    df.drop('Day', axis=1, inplace=True)

    logging.info('Encode StateHoliday')
    cat_cols = ['StateHoliday']
    df, le = encode_cat_cols(df, cat_cols)

    logging.info(f'Save label encoder to {le_file}.pkl')
    general_utils.save_pickle_file(le_file, le)
    return df, le


def inverselogtransform_sales_df(df):
    df['Sales'] = np.exp(df['Sales']) - 1
    # df['Customers'] = np.exp(df['Customers']) - 1
    return df


def logtransform_sales_df(df):
    df['Sales'] = np.log1p(df['Sales'])
    df['Customers'] = np.log1p(df['Customers'])
    return df


def rescale_data(scale_map, df, columns=['predictions', 'y_sequence']):
    rescaled_data = pd.DataFrame()
    logging.info(f'Start rescaling')
    for store_id, store_data in tqdm(df.groupby('Store', as_index=False)):
        mean = scale_map['Sales'][store_id]['mean']
        std = scale_map['Sales'][store_id]['std']
        for col in columns:
            store_data[col] = store_data[col].apply(lambda x: (np.array(x) * std) + mean)
            store_data = store_data[col].apply(lambda x: np.exp(x) - 1)
        rescaled_data = pd.concat([rescaled_data, store_data], ignore_index=True)
    return rescaled_data



def scale_data(df, scalemap_file_name, scaled_data_filename, cols_to_scale, mode='val', val_date=None):
    logging.info('Log Transform Sales and Customers')
    df = logtransform_sales_df(df)

    logging.info('Select records for scaler fitting')
    if mode == 'val':
        train = df[df['Date'] < val_date]
    else:
        train = df

    scale_map = defaultdict(dict)
    scaled_data = pd.DataFrame()
    logging.info(f'Start scaling time-dependant features {cols_to_scale["td"]}')
    for store_id, store_data in tqdm(df.groupby('Store', as_index=False)):
        store_subset = train.loc[train['Store'] == store_id]
        for col in cols_to_scale['td']:
            mean = store_subset[col].mean()
            std = store_subset[col].std()
            store_data.loc[:, col] = (store_data[col] - mean) / std
            scale_map[col][store_id] = {'mean': mean, 'std': std}
            store_data[f'{col}_mean'] = mean
        scaled_data = pd.concat([scaled_data, store_data], ignore_index=True)

    logging.info(f'Start scaling time-independant features {cols_to_scale["ti"]}')
    for col in cols_to_scale['ti']:
        mean = train[col].mean()
        std = train[col].std()
        scaled_data.loc[:, col] = (scaled_data[col] - mean) / std
        scaled_data[f'{col}_mean'] = mean
        scale_map[col]['mean'] = mean
        scale_map[col]['std'] = std

    logging.info('Convert data formats to reduce memory usage')
    scaled_data = general_utils.reduce_mem_usage(scaled_data)

    logging.info('Save pickle files')
    general_utils.save_pickle_file(scalemap_file_name, scale_map)
    general_utils.save_pickle_file(scaled_data_filename, scaled_data)

    return scaled_data, scale_map


"""
Class for pytorch dataset
"""


class StoreDataset(Dataset):
    """
    Characterizes a Dataset for PyTorch
    """
    def __init__(self, cat_columns=[], num_columns=[], embed_vector_size=None, decoder_input=False,
                 cat_columns_to_decoder=False,
                 ohe_cat_columns=False):
        super().__init__()
        logging.info('Create Dataset object')
        self.sequence_data = None
        self.cat_columns = cat_columns
        self.num_columns = num_columns
        self.cat_classes = {}
        self.cat_embed_shape = []
        self.cat_embed_vector_size = embed_vector_size if embed_vector_size is not None else {}
        self.pass_decoder_input = decoder_input
        self.ohe_cat_columns = ohe_cat_columns
        self.cat_columns_to_decoder = cat_columns_to_decoder

    def get_embedding_shape(self):
        return self.cat_embed_shape

    def load_sequence_data(self, processed_data):
        logging.info('Load data')
        self.sequence_data = processed_data

    def process_cat_columns(self):
        """
        :return: list of tuples, where each tuple represents
        a pair of total and the embedding dimension of a categorical variable
        """
        cat_dims = [int(self.sequence_data[col].nunique()) for col in self.cat_columns]
        self.cat_embed_shape = [(x, min(self.cat_embed_vector_size, (x + 1) // 2)) for x in cat_dims]

    def __len__(self):
        """
        Denotes the total number of samples.
        """
        return len(self.sequence_data)

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        row = self.sequence_data.iloc[[idx]]  # <class 'pandas.core.frame.DataFrame'>
        x_inputs = [torch.tensor(row['x_sequence'].values[0], dtype=torch.float32)]
        y = torch.tensor(row['y_sequence'].values[0][:, 0], dtype=torch.float32)
        decoder_input = torch.empty(row['y_sequence'].values[0].shape[0], dtype=torch.float32)
        if self.pass_decoder_input:
            # pass lag features to decoder
            lag_input = torch.tensor(row['y_sequence'].values[0][:, 1:], dtype=torch.float32)
            decoder_input = torch.cat((decoder_input, lag_input))

        if len(self.num_columns) > 0:
            for col in self.num_columns:
                # add column vector of numerical feature near x_sequence and decoder_input
                # x_sequence consists of time-dependant + additional numerical columns
                # decoder_input consists of lag features + additional numerical columns
                num_tensor = torch.tensor([row[col].values[0]], dtype=torch.float32)
                x_inputs[0] = torch.cat((x_inputs[0], num_tensor.repeat(x_inputs[0].size(0)).unsqueeze(1)), axis=1)
                if self.pass_decoder_input:
                    decoder_input = torch.cat((decoder_input, num_tensor.repeat(decoder_input.size(0)).unsqueeze(1)), axis=1)

        if len(self.cat_columns) > 0:
            if self.ohe_cat_columns:
                for ci, (num_classes, _) in enumerate(self.cat_embed_shape):
                    col_tensor = torch.zeros(num_classes, dtype=torch.float32)
                    col_tensor[row[self.cat_columns[ci]].values[0]] = 1.0
                    col_tensor_x = col_tensor.repeat(x_inputs[0].size(0), 1)
                    x_inputs[0] = torch.cat((x_inputs[0], col_tensor_x), axis=1)
                    if self.pass_decoder_input and self.cat_columns_to_decoder:
                        col_tensor_y = col_tensor.repeat(decoder_input.size(0), 1)
                        decoder_input = torch.cat((decoder_input, col_tensor_y), axis=1)
            else:
                cat_tensor = torch.tensor(
                    [row[col].values[0] for col in self.cat_columns],
                    dtype=torch.long
                )
                x_inputs.append(cat_tensor)
        if self.pass_decoder_input:
            x_inputs.append(decoder_input)
        if len(x_inputs) > 1:
            return tuple(x_inputs), y
        return x_inputs[0], y
