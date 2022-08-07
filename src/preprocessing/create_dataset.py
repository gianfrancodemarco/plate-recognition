import sys
import logging.config
logging.config.fileConfig('logging.conf')

sys.path.append('../..')

from dataset_creator import DatasetCreator
DatasetCreator().create_dataset()