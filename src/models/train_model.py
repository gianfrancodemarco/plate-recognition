import os

import mlflow
import pandas as pd
import tensorflow as tf
from src import utils
from src.features.dataset import (AugmentedImagesDatasetGenerator,
                                  configure_for_performance)
from src.models.get_model import get_model
from src.models.model_trainer import train_model

DATASETS_BASE = os.path.join(utils.DATA_PATH, "processed")


def get_dataset(split: str) -> tf.data.Dataset:

    annotations_path = os.path.join(DATASETS_BASE, split, "annotations.csv")
    annotations = pd.read_csv(annotations_path)

    images_path = os.path.join(DATASETS_BASE, split, "images")

    dataset_generator = AugmentedImagesDatasetGenerator(
        images_path=images_path,
        annotations=annotations
    )

    dataset = tf.data.Dataset.from_generator(
        dataset_generator.get_image,
        output_signature=(tf.TensorSpec(shape=(256, 256, 3)), tf.TensorSpec(shape=(4, )))
    )

    dataset = configure_for_performance(dataset)

    return dataset


if __name__ == "__main__":

    MODEL_NAME = "third_model"
    MODEL_VERSION = 1

    train_set = get_dataset("train")
    validation_set = get_dataset("validation")

    # Fetches the corresponding model from MLFlow artifacts storage
    # If it doesn't exists, creates a new model
    model = get_model(
        dropout=0.3,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION
    )

    with mlflow.start_run():

        # Automatically capture the model's parameters, metrics, artifacts,
        # and source code with the `autolog()` function
        mlflow.tensorflow.autolog()

        train_model(
            model=model,
            dataset=train_set,
            validation_dataset=validation_set,
            model_name=MODEL_NAME
        )
