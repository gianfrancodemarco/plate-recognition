import itertools
import os
from typing import Tuple

import cv2
import numpy as np
import tensorflow.keras as keras
from keras.callbacks import Callback
from keras.layers import (AveragePooling2D, BatchNormalization, Conv2D, Dense,
                          Dropout, Flatten, Input, LeakyReLU, MaxPooling2D)
from keras.metrics import MeanSquaredError, RootMeanSquaredError
from keras.models import Sequential, load_model
from shapely.geometry import Polygon
from src import utils
from src.data.annotations.coco_annotations_manager import \
    CocoAnnotationsManager
from src.visualization.image_visualizer import show_image
from src.features.dataset import ImagesDatasetGenerator
import tensorflow as tf

model = load_model('first_model.h5')

validation_annotations_path = os.path.join(utils.DATA_PATH, 'annotations', 'validation_set_annotations.json')
validation_annotations_manager = CocoAnnotationsManager()
validation_annotations_manager.load_annotations(validation_annotations_path)
validation_images_base_path = os.path.join(utils.DATA_PATH, 'processed', 'validation')
validation_images_paths =  [os.path.join(validation_images_base_path, image['file_name']) for image in validation_annotations_manager.get_images()]
validation_dataset_generator = ImagesDatasetGenerator(
    images_paths=validation_images_paths[:100],
    annotations=validation_annotations_manager.get_flattened_segmentations(),
    pad_annotations=50
)

validation_dataset = tf.data.Dataset.from_generator(
    validation_dataset_generator.get_image,
    output_signature=(tf.TensorSpec(shape=(512, 512, 3)), tf.TensorSpec(shape=(50, )))
)

asd = validation_dataset.take(1)
polygons = model.predict(np.array([list(asd)[0][0]]))
polygons = np.array_split([el/(1500/512) if el > 0 else 0 for el in polygons[0]], 5)
polygons = [Polygon(np.array_split(polygon, 5)) for polygon in polygons]
show_image(list(asd)[0][0], polygons)