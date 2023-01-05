import logging
import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from src.data.image_preprocessing import random_image_augmentation


def configure_for_performance(ds: tf.data.Dataset):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(16)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


class ImagesDatasetGenerator():

    def __init__(
        self,
        annotations: pd.DataFrame,
        images_path: str
    ):
        self.annotations = annotations.values.tolist()
        self.images_path = images_path

    def get_image(self) -> (np.ndarray):
        for sample in self.annotations:
            try:
                image_name = sample[0]
                image_path = os.path.join(self.images_path, image_name)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                annotation = sample[1:-1]
                yield image, annotation
            except Exception as e:
                logging.error("Error retrieving dataset image.")
                logging.exception(e)                
                logging.info(f"Image path: {image_path}, annotation: {annotation}")

    def image_transformation(image):
        image = cv2.cvtColor(cv2.COLOR_BGR2RGB)
        return image

class AugmentedImagesDatasetGenerator(ImagesDatasetGenerator):

    def image_transformation(image):
        image = cv2.cvtColor(cv2.COLOR_BGR2RGB)
        image = random_image_augmentation(image)
        return image