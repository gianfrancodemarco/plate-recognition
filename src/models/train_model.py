import logging

import dvc.api
import mlflow
from src.features.dataset import get_dataset
from src.features.dataset_generator import ImageDatasetType
from src.models.build_model import build_model
from src.models.fetch_model import fetch_model
from src.models.train_callbacks import (EarlyStoppingByLossVal,
                                        SaveModelMLFlowCallback)
from src.utils import set_random_states

params = dvc.api.params_show()

assert "random_state" in params, "Required param random_state" 
set_random_states(params["random_state"])

assert "train" in params, "Required param train"
train_params = params["train"]

model_params = train_params.get("model")
assert "model_name" in model_params, "Required param model.model_name"
assert "model_version" in model_params, "Required param model.model_version"
model_name = model_params["model_name"]
model_version = model_params["model_version"]

fit_params = train_params.get("fit")
assert "epochs" in fit_params, "Required param fit.epochs"
epochs = fit_params["epochs"]


if __name__ == "__main__":

    train_set = get_dataset("train", dataset_generator_type=ImageDatasetType.BboxAugmentedImagesDatasetGenerator)
    validation_set = get_dataset("validation", dataset_generator_type=ImageDatasetType.BboxImagesDatasetGenerator)
   
    try:
        model = fetch_model(model_name=model_name, model_version=model_version)
    except Exception as e:
        logging.exception(e)
        model = build_model(**model_params)

    with mlflow.start_run():

        mlflow.log_params(params)

        mlflow.tensorflow.autolog(
            log_input_examples=True,
            log_models=True
        )

        callbacks = [
            EarlyStoppingByLossVal(monitor='loss', value=1, verbose=1),
            SaveModelMLFlowCallback(
                model_name=model_name
            )
        ]

        logging.info(f"Training the model for {epochs} epochs")

        model.fit(
            x=train_set,
            validation_data=validation_set,
            batch_size=16,
            verbose=1,
            validation_split=validation_set,
            callbacks=callbacks,
            epochs=epochs
        )
