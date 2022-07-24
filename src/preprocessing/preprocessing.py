import logging.config
logging.config.fileConfig('logging.conf')

from preprocessing_kaggle import KaggleDatasetPreprocessor
from preprocessing_baza_slika import BazaSlikaDatasetPreprocessor

KaggleDatasetPreprocessor().run_preprocessing()
BazaSlikaDatasetPreprocessor().run_preprocessing()