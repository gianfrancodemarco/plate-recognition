import json
import logging
import os

import dvc.api
import mlflow
from src import utils
from src.features.dataset import get_dataset
from src.features.dataset_generator import ImageDatasetType
from src.models.build_model import build_model
from src.models.fetch_model import fetch_model
from src.models.train_callbacks import (EarlyStoppingByLossVal,
                                        SaveModelMLFlowCallback)
from src.pipeline.param_parser import ParamParser

TRAIN_REPORTS_PATH = os.path.join(utils.REPORTS_PATH, "train")

params_dict = dvc.api.params_show()
params = ParamParser().parse(params_dict)
utils.set_random_states(params.random_state)

if __name__ == "__main__":

    train_set = get_dataset(
        "train", dataset_generator_type=ImageDatasetType.BboxAugmentedImagesDatasetGenerator)
    validation_set = get_dataset(
        "validation", dataset_generator_type=ImageDatasetType.BboxImagesDatasetGenerator)

    try:
        model = fetch_model(model_name=params.train.model.model_name,
                            model_version=params.train.model.model_version)
    except Exception as e:
        logging.exception(e)
        model = build_model(
            dropout=params.train.model.dropout,
            cnn_blocks=params.train.model.cnn_blocks,
            filters_num=params.train.model.filters_num,
            filters_kernel_size=params.train.model.filters_kernel_size
        )

    with mlflow.start_run():
        mlflow.log_params(params.__dict__)
        mlflow.tensorflow.autolog(
            log_input_examples=True,
            log_models=True
        )

        callbacks = [
            EarlyStoppingByLossVal(monitor='loss', value=1, verbose=1),
            SaveModelMLFlowCallback(
                model_name=params.train.model.model_name
            )
        ]

        logging.info(f"Training the model for {params.train.fit.epochs} epochs")

        model.fit(
            x=train_set,
            validation_data=validation_set,
            batch_size=16,
            verbose=1,
            validation_split=validation_set,
            callbacks=callbacks,
            epochs=params.train.fit.epochs
        )


        output_path = os.path.join(TRAIN_REPORTS_PATH, "history.json")
        with open(output_path, "w") as f:
            f.write(json.dumps(model.history.history, indent=4))
