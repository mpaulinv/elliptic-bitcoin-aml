# Simple script to download the Elliptic dataset from Kaggle
# Make sure to have Kaggle API installed and configured

import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_elliptic_dataset():
    """
    Downloads the Elliptic dataset from Kaggle and saves it to the 'data/' folder.
    """
    # Define the default download path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get the project root directory
    data_dir = os.path.join(base_dir, "data")

    # Ensure the data directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Set up Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Download the dataset
    print(f"Downloading the Elliptic dataset to: {data_dir}")
    api.dataset_download_files('ellipticco/elliptic-data-set', path=data_dir, unzip=True)

    print("Downloaded Elliptic dataset into the 'data/' folder.")

if __name__ == "__main__":
    download_elliptic_dataset()