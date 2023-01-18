import os

import pandas as pd
import tensorflow as tf
from src import utils
from src.features.dataset_generator import (ImageDatasetType,
                                            get_dataset_generator)

DATASETS_BASE = os.path.join(utils.DATA_PATH, "processed")


def configure_for_performance(ds: tf.data.Dataset):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(16)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def get_dataset(
    split: str,
    dataset_generator_type: ImageDatasetType = ImageDatasetType.ImagesDatasetGenerator
) -> tf.data.Dataset:

    annotations_path = os.path.join(DATASETS_BASE, split, "annotations.csv")
    annotations = pd.read_csv(annotations_path)

    images_path = os.path.join(DATASETS_BASE, split, "images")

    dataset_generator_class = get_dataset_generator(dataset_generator_type)
    dataset_generator = dataset_generator_class(
        images_path=images_path,
        annotations=annotations
    )

    dataset = tf.data.Dataset.from_generator(
        dataset_generator.get_image,
        output_signature=(tf.TensorSpec(shape=(256, 256, 3)), tf.TensorSpec(shape=(4, )))
    )

    dataset = configure_for_performance(dataset)

    return dataset
