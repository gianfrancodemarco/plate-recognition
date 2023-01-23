import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from src.data.image_preprocessing import random_image_augmentation


class ImageDatasetType(Enum):
    PLATE_IMAGES_DATASET_GENERATOR = 'PLATE_IMAGES_DATASET_GENERATOR'
    BBOX_IMAGES_DATASET_GENERATOR = 'BBOX_IMAGES_DATASET_GENERATOR'
    BBOX_AUGMENTED_IMAGES_DATASET_GENERATOR = 'BBOX_AUGMENTED_IMAGES_DATASET_GENERATOR'

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
            except Exception as exc:
                logging.error("Error retrieving dataset image.")
                logging.exception(exc)
                logging.info(f"Image path: {image_path}, annotation: {annotation}")

    @abstractmethod
    def image_transformation(self, image: np.ndarray):
        pass

    @abstractmethod
    def get_annotation(self, annotation):
        pass

class BboxImagesDatasetGenerator(ImagesDatasetGenerator):

    def get_annotation(self, annotation) -> Tuple[np.ndarray, np.ndarray]:
        return annotation[1:-1]

    def image_transformation(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

class BboxAugmentedImagesDatasetGenerator(ImagesDatasetGenerator):

    def get_annotation(self, annotation) -> Tuple[np.ndarray, np.ndarray]:
        return annotation[1:-1]

    def image_transformation(self, image):
        return random_image_augmentation(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

class PlateImagesDatasetGenerator(ImagesDatasetGenerator):

    def get_annotation(self, annotation) -> Tuple[np.ndarray, np.ndarray]:
        return annotation[-1]

    def image_transformation(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = random_image_augmentation(image)
        return image


def get_dataset_generator(dataset_generator_type: ImageDatasetType) -> ImagesDatasetGenerator:
    if dataset_generator_type == ImageDatasetType.BBOX_IMAGES_DATASET_GENERATOR:
        return BboxImagesDatasetGenerator
    if dataset_generator_type == ImageDatasetType.BBOX_AUGMENTED_IMAGES_DATASET_GENERATOR:
        return BboxAugmentedImagesDatasetGenerator
    if dataset_generator_type == ImageDatasetType.PLATE_IMAGES_DATASET_GENERATOR:
        return PlateImagesDatasetGenerator

def get_model_output_signature(dataset_generator_type: ImageDatasetType):
    if dataset_generator_type == ImageDatasetType.BBOX_IMAGES_DATASET_GENERATOR:
        return tf.TensorSpec(shape=(256, 256, 3)), tf.TensorSpec(shape=(4, ))
    if dataset_generator_type == ImageDatasetType.BBOX_AUGMENTED_IMAGES_DATASET_GENERATOR:
        return tf.TensorSpec(shape=(256, 256, 3)), tf.TensorSpec(shape=(4, ))
    if dataset_generator_type == ImageDatasetType.PLATE_IMAGES_DATASET_GENERATOR:
        return tf.TensorSpec(shape=(256, 256, 3)), tf.TensorSpec(shape=(), dtype=tf.string)
