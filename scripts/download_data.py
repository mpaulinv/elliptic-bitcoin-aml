#simple script to download the Elliptic dataset from Kaggle
# Make sure to have Kaggle API installed and configured

import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Set up Kaggle API
api = KaggleApi()
api.authenticate()

# Download to "data/" folder
api.dataset_download_files('ellipticco/elliptic-data-set', path='C:/Users/mario/elliptic-bitcoin-aml/data/', unzip=True)

print("Downloaded Elliptic dataset into the 'data/' folder.")
