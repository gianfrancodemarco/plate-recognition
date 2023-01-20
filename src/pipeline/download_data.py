import logging
import os
import zipfile

from src import utils
from src.data.misc import fetch_or_resume

RAW_PATH = os.path.join(utils.DATA_PATH, "raw")
DATASET_PATH = os.path.join(RAW_PATH, "plate-recognition-dataset.zip")


def download_dataset():
    DATA_URL = os.getenv(
        "DATASET_URL", "https://drive.google.com/uc?id=1bTxEOmOtIzHPV8N43ZnJTyVu7lonIP1o&export=download")

    logging.info("Downloading dataset...")
    fetch_or_resume(DATA_URL, DATASET_PATH)

    logging.info("Download complete. Extracting zip file...")
    with zipfile.ZipFile(DATASET_PATH, 'r') as zip_ref:
        zip_ref.extractall(RAW_PATH)

    logging.info("Zip file extracted. Removing zip file...")
    os.remove(DATASET_PATH)

    logging.info("Zip file removed.")


if __name__ == "__main__":
    download_dataset()