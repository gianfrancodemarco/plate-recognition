import os

import mlflow
import mlflow.sklearn
import tensorflow as tf
from src import utils
from src.data.annotations.coco_annotations_manager import \
    CocoAnnotationsManager
from src.features.dataset import ImagesDatasetGenerator
from src.models.get_model import build_model
from src.models.model_trainer import train_model


def get_datasets():
    datasets = []
    for split in ['train', 'validation']:
        annotations_path = os.path.join(utils.DATA_PATH, 'annotations', f'{split}_set_annotations.json')
        annotations_manager = CocoAnnotationsManager()
        annotations_manager.load_annotations(annotations_path)
        images_base_path = os.path.join(utils.DATA_PATH, 'processed', split)
        images_paths =  [os.path.join(images_base_path, image['file_name']) for image in annotations_manager.get_images()]
        dataset_generator = ImagesDatasetGenerator(
            images_paths=images_paths,
            annotations=annotations_manager.get_flattened_segmentations(),
            pad_annotations=50
        )

        dataset = tf.data.Dataset.from_generator(
            dataset_generator.get_image,
            output_signature=(tf.TensorSpec(shape=(512, 512, 3)), tf.TensorSpec(shape=(50, )))
        )

        dataset = configure_for_performance(dataset)
        datasets.append(dataset)
    
    return datasets

def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(16)
  ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
  return ds

if __name__ == "__main__":

    MODEL_NAME = "third_model"
    MODEL_VERSION = 1

    train_set, validation_set = get_datasets()

    # Fetches the corresponding model from MLFlow artifacts storage
    # If it doesn't exists, creates a new model
    model = build_model(
        dropout=0.3,
        model_name = MODEL_NAME,
        model_version = MODEL_VERSION
    )
    
    with mlflow.start_run():
        
        # Automatically capture the model's parameters, metrics, artifacts,
        # and source code with the `autolog()` function
        mlflow.tensorflow.autolog()

        train_model(
            model = model, 
            dataset = train_set,
            validation_dataset=validation_set,
            model_name=MODEL_NAME
        )
