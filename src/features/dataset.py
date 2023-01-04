import logging
import os

import cv2
import numpy as np
import pandas as pd

class ImagesDatasetGenerator():

    def __init__(
        self, 
        annotations: pd.DataFrame,
        images_path: str
    ):
        self.annotations = annotations.values.tolist()
        self.images_path = images_path
        
    def get_image(self):
        for sample in self.annotations:
            try:
                image_name = str(sample[0]) + '.jpg'
                image_path = os.path.join(self.images_path, image_name)
                image = cv2.imread(image_path)
                image = cv2.resize(image, (512, 512))
                annotation = sample[1:]
                yield image, annotation
            except:
                logging.error(image_path)
                logging.error(annotation)
