import os

import pandas as pd
import tensorflow as tf
from src.features.dataset_generator import (ImageDatasetType,
                                            get_dataset_generator,
                                            get_model_output_signature)


def configure_for_performance(dataset: tf.data.Dataset, batch_size: int = 16, shuffle: bool = True):
    #dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def get_dataset(
    annotations_path: str,
    dataset_generator_type: ImageDatasetType = ImageDatasetType.BBOX_IMAGES_DATASET_GENERATOR,
    batch_size: int = 16,
    shuffle: bool = True
) -> tf.data.Dataset:

    annotations = pd.read_csv(annotations_path)

    base_path = os.path.dirname(annotations_path)
    images_path = os.path.join(base_path, "images")

    dataset_generator_class = get_dataset_generator(dataset_generator_type)
    dataset_generator = dataset_generator_class(
        images_path=images_path,
        annotations=annotations
    )

    output_signature = get_model_output_signature(dataset_generator_type)
    dataset = tf.data.Dataset.from_generator(
        dataset_generator.get_sample,
        output_signature=output_signature
    )

    dataset = configure_for_performance(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataset
