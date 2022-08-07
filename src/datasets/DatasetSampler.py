import os
import sys
import random
import logging.config

import pandas as pd

from src.datasets.AnnotatedImageVisualizer import AnnotatedImageVisualizer

logging.config.fileConfig('logging.conf')

sys.path.append('../..')

dir_path = os.path.dirname(os.path.realpath(__file__))


class DatasetSampler:
    DATASET_PATH = os.path.join(dir_path, '..', '..', 'final_dataset')
    IMAGES_PATH = os.path.join(DATASET_PATH, 'images')

    def sample_dataset(self):

        images = os.listdir(self.IMAGES_PATH)
        image = random.choice(images)
        image_name = int(image.split('.')[0])
        annotations = pd.read_csv(os.path.join(self.DATASET_PATH, 'annotations.csv'))

        row = annotations[annotations['name'] == image_name]

        row_annotations = None
        if not row.empty:
            row = row.iloc[0]
            row_annotations = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]

        AnnotatedImageVisualizer().show_image(os.path.join(self.IMAGES_PATH, image), row_annotations)