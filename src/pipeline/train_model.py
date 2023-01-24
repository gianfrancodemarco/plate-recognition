import logging
import os

import dvc.api
from src import utils
from src.features.dataset import get_dataset
from src.features.dataset_generator import ImageDatasetType
from src.models.fetch_model import fetch_model
from src.models.mlflow_model_trainer import MLFlowModelTrainer
from src.models.model_builder import ModelBuilder
from src.pipeline.param_parser import ParamParser

DATASETS_BASE = os.path.join(utils.DATA_PATH, "processed")
TRAIN_REPORTS_PATH = os.path.join(utils.REPORTS_PATH, "train")

params_dict = dvc.api.params_show()
params = ParamParser().parse(params_dict)
utils.set_random_states(params.random_state)

if __name__ == "__main__":

    dataset_generator_type = ImageDatasetType.BBOX_IMAGES_DATASET_GENERATOR
    if params.augmentation:
        dataset_generator_type = ImageDatasetType.BBOX_AUGMENTED_IMAGES_DATASET_GENERATOR

    train_set = get_dataset(
        annotations_path=os.path.join(DATASETS_BASE, "train", "annotations.csv"),
        dataset_generator_type=dataset_generator_type
    )
    
    validation_set = get_dataset(
        annotations_path=os.path.join(DATASETS_BASE, "validation", "annotations.csv"),
        dataset_generator_type=dataset_generator_type
    )

    try:
        model = fetch_model(model_name=params.train.model.model_name,
                            model_version=params.train.model.model_version)
    except Exception as e:
        logging.exception(e)

        model = ModelBuilder(
            dropout=params.train.model.dropout,
            cnn_blocks=params.train.model.cnn_blocks,
            filters_num=params.train.model.filters_num,
            filters_kernel_size=params.train.model.filters_kernel_size,
            optimizer_name=params.train.model.optimizer_name,
            learning_rate=params.train.model.learning_rate
        ).build()

    model_trainer = MLFlowModelTrainer(
        epochs=params.train.fit.epochs,
        model_name=params.train.model.model_name,
        train_data=train_set,
        validation_data=validation_set
    )

    results_path = os.path.join(TRAIN_REPORTS_PATH, "history.json")
    model_trainer.train(
        model,
        params_to_log=params.__dict__,
        results_path=results_path
    )
