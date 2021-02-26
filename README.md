# Takehome task

Project contains a solution of a multisite multivariate multistep time forecasting task.\
Main points:
* Encoder-Decoder RNN model
* Multivariate 120 length input sequence, 47 length 1-D output sequence

To make a solution, you can run either Local_version.ipynb or FullPipelineColab.ipynb with an additional requirements installation.

To run a local solution:
python 3.7.8 for pickle files
1. `pip install -r requirements.txt`
2. run Local_version.ipynb


To run a Google Colab notebook:
* Upload FullPipelineColab.ipynb to Google Colab 
1. Create a folder for a project on Google Drive
2. Go to this folder and choose Open with Google Colab by clicking on a right button
3. Mount to your Google Drive disk
4. Init a git repository, pull this repo 
5. Move to a folder with this repo. Create a folder largefiles in the root of a repo.
6. Move uploaded FullPipelineColab.ipynb to the folder with this repo and run it

Files with a pretrained model and modified data from colab notebook could be placed in the root of a repo directory.
They are not needed for the coldstart.

They can be downloaded by the following links: 

largefiles/:
https://drive.google.com/drive/folders/17-0oupaG31DIZwbroBDe2nn53Xgxbrmb?usp=sharing

models/:
https://drive.google.com/drive/folders/11Md6b9jrnWj4vOAXZa94FJ551RI0dSZd?usp=sharing

runs/:
https://drive.google.com/drive/folders/11GRgz7S3cRBlrww6uBcPqg3JpX1Euf0W?usp=sharing
