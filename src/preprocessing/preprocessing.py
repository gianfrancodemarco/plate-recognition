import sys
import logging.config
logging.config.fileConfig('logging.conf')

sys.path.append('../..')

from preprocessing_kaggle import KaggleDatasetPreprocessor
from preprocessing_baza_slika import BazaSlikaDatasetPreprocessor
from preprocessing_license_plate_recognition import LicensePlateRecognitionDatasetPreprocessor

KaggleDatasetPreprocessor().run_preprocessing()
BazaSlikaDatasetPreprocessor().run_preprocessing()
LicensePlateRecognitionDatasetPreprocessor().run_preprocessing()
