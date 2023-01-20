import logging
import os
from abc import abstractclassmethod, ABC
from enum import Enum
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from src.data.image_preprocessing import random_image_augmentation


class ImageDatasetType(Enum):
    PlateImagesDatasetGenerator = 'PlateImagesDatasetGenerator'
    BboxImagesDatasetGenerator = 'BboxImagesDatasetGenerator'
    BboxAugmentedImagesDatasetGenerator = 'BboxAugmentedImagesDatasetGenerator'

class ImagesDatasetGenerator(ABC):
    """
    A base class for generating Tensorflow datasets for images

    Subclasses must implement the logic for transforming the images as needed and to retrieve the correct annotation from the dataframe
    """

    def __init__(
        self,
        annotations: pd.DataFrame,
        images_path: str
    ):
        self.annotations = annotations.values.tolist()
        self.images_path = images_path

    def get_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        for sample in self.annotations:
            try:
                image_name = sample[0]
                image_path = os.path.join(self.images_path, image_name)
                image = cv2.imread(image_path)
                image = self.image_transformation(image)
                annotation = self.get_annotation(sample)
                yield image, annotation
            except Exception as e:
                logging.error("Error retrieving dataset image.")
                logging.exception(e)
                logging.info(f"Image path: {image_path}, annotation: {annotation}")

    @abstractclassmethod
    def image_transformation(cls, image: np.ndarray):
        pass

    @abstractclassmethod
    def get_annotation(cls, annotation):
        pass

class BboxImagesDatasetGenerator(ImagesDatasetGenerator):

    def get_annotation(cls, annotation) -> Tuple[np.ndarray, np.ndarray]:
        return annotation[1:-1]
    
    def image_transformation(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

class BboxAugmentedImagesDatasetGenerator(ImagesDatasetGenerator):
    
    def get_annotation(cls, annotation) -> Tuple[np.ndarray, np.ndarray]:
        return annotation[1:-1]
    
    def image_transformation(self, image):
        return random_image_augmentation(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

class PlateImagesDatasetGenerator(ImagesDatasetGenerator):
    
    def get_annotation(cls, annotation) -> Tuple[np.ndarray, np.ndarray]:
        return annotation[-1]
    
    def image_transformation(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = random_image_augmentation(image)
        return image


def get_dataset_generator(dataset_generator_type: ImageDatasetType):
    if dataset_generator_type == ImageDatasetType.BboxImagesDatasetGenerator:
        return BboxImagesDatasetGenerator
    elif dataset_generator_type == ImageDatasetType.BboxAugmentedImagesDatasetGenerator:
        return BboxAugmentedImagesDatasetGenerator
    elif dataset_generator_type == ImageDatasetType.PlateImagesDatasetGenerator:
        return PlateImagesDatasetGenerator

def get_model_output_signature(dataset_generator_type: ImageDatasetType):
    if dataset_generator_type == ImageDatasetType.BboxImagesDatasetGenerator:
        return tf.TensorSpec(shape=(256, 256, 3)), tf.TensorSpec(shape=(4, ))
    elif dataset_generator_type == ImageDatasetType.BboxAugmentedImagesDatasetGenerator:
        return tf.TensorSpec(shape=(256, 256, 3)), tf.TensorSpec(shape=(4, ))
    elif dataset_generator_type == ImageDatasetType.PlateImagesDatasetGenerator:
        return tf.TensorSpec(shape=(256, 256, 3)), tf.TensorSpec(shape=(), dtype=tf.string)

