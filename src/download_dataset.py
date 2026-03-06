import os
import requests
import zipfile

# Dataset URL
DATASET_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"

# Paths
DOWNLOAD_PATH = "dataset/maestro.zip"
EXTRACT_PATH = "dataset"

def download_dataset():

    os.makedirs("dataset", exist_ok=True)

    print("Downloading MAESTRO dataset...")

    response = requests.get(DATASET_URL, stream=True)

    with open(DOWNLOAD_PATH, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)

    print("Download complete!")

def extract_dataset():

    print("Extracting dataset...")

    with zipfile.ZipFile(DOWNLOAD_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)

    print("Extraction complete!")

if __name__ == "__main__":

    download_dataset()
    extract_dataset()