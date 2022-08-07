import logging
import os
import glob
import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
dir_path = os.path.dirname(os.path.realpath(__file__))


class DatasetLoader:

    KAGGLE_PATH = os.path.join(dir_path, '..', '..', 'resources', 'kaggle')
    KAGGLE_IMAGES = os.path.join(KAGGLE_PATH, 'images')
    KAGGLE_ANNOTATIONS = os.path.join(KAGGLE_PATH, 'annotations.csv')

    def __init__(self):

        self.kaggle_X = []
        self.kaggle_y = []

    def load_kaggle_dataset(self):

        logger.info("Loading kaggle dataset")

        data_path = os.path.join(self.KAGGLE_IMAGES, '*.png')
        files = glob.glob(data_path)
        files.sort()  # We sort the images in alphabetical order to match them to the xml files containing the
        # annotations of the bounding boxes

        for f1 in files:
            img = cv2.imread(f1)
            self.kaggle_X.append(np.array(img))

        logger.info("Loaded images")

        annotations = pd.read_csv(self.KAGGLE_ANNOTATIONS)
        annotations = annotations.drop(annotations.columns[[0]], axis=1)
        self.kaggle_y = annotations.values.tolist()

        # Normalize data
        # Transforming in array
        self.kaggle_X = np.array(self.kaggle_X)#.astype(np.float16)
        self.kaggle_y = np.array(self.kaggle_y)#.astype(np.float16)

        # Renormalisation
        #self.kaggle_X = self.kaggle_X / 255
        #self.kaggle_y = self.kaggle_y / 255

        #logger.info("Loaded annotations")
