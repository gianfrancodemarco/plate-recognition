import sys
import logging.config
logging.config.fileConfig('logging.conf')

sys.path.append('..')

from models import plate_detection

# from datasets.DatasetSampler import DatasetSampler
#
# for i in range(10):
#     DatasetSampler().sample_dataset()