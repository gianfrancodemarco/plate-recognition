import logging
import os
from enum import Enum
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from src import utils
from src.data.image_preprocessing import random_image_augmentation


class ImageDatasetType(Enum):
    ImagesDatasetGenerator = 'ImagesDatasetGenerator'
    AugmentedImagesDatasetGenerator = 'AugmentedImagesDatasetGenerator'


class ImagesDatasetGenerator():

    def __init__(
        self,
        annotations: pd.DataFrame,
        images_path: str
    ):
        self.annotations = annotations.values.tolist()
        self.images_path = images_path

    def get_image(self) -> Tuple[np.ndarray, np.ndarray]:
        for sample in self.annotations:
            try:
                image_name = sample[0]
                image_path = os.path.join(self.images_path, image_name)
                image = cv2.imread(image_path)
                image = self.image_transformation(image)

                # We want the annotations in the form: [y_min, x_min, y_max, x_max]
                # to be able to use GIoU loss
                # https://github.com/tensorflow/addons/blob/master/tensorflow_addons/losses/giou_loss.py

                annotation = [sample[2], sample[1], sample[4], sample[3]]
                yield image, annotation
            except Exception as e:
                logging.error("Error retrieving dataset image.")
                logging.exception(e)
                logging.info(f"Image path: {image_path}, annotation: {annotation}")

    def image_transformation(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


class AugmentedImagesDatasetGenerator(ImagesDatasetGenerator):

    def image_transformation(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = random_image_augmentation(image)
        return image


def get_dataset_generator(dataset_generator_type: ImageDatasetType):
    if dataset_generator_type == ImageDatasetType.ImagesDatasetGenerator:
        return ImagesDatasetGenerator
    elif dataset_generator_type == ImageDatasetType.AugmentedImagesDatasetGenerator:
        return AugmentedImagesDatasetGenerator
