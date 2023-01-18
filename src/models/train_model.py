import logging

import dvc.api
import mlflow
from src.features.dataset import get_dataset
from src.models.fetch_model import fetch_model
from src.models.build_model import build_model
from src.models.train_callbacks import EarlyStoppingByLossVal, SaveModelMLFlowCallback
from src.features.dataset_generator import ImageDatasetType

params = dvc.api.params_show()
assert "train" in params, "Required param train"
train_params = params["train"]

model_params = train_params.get("model")
assert "model_name" in model_params, "Required param model.model_name"
assert "model_version" in model_params, "Required param model.model_version"
model_name = model_params["model_name"]
model_version = model_params["model_version"]

fit_params = train_params.get("fit")
assert "epochs" in fit_params, "Required param fit.epochs"
assert "save_every_n_epochs" in fit_params, "Required param fit.save_every_n_epochs"
epochs = fit_params["epochs"]
save_every_n_epochs = fit_params["save_every_n_epochs"]


if __name__ == "__main__":

    train_set = get_dataset("train", dataset_generator_type=ImageDatasetType.AugmentedImagesDatasetGenerator)
    validation_set = get_dataset("validation", dataset_generator_type=ImageDatasetType.ImagesDatasetGenerator)
   
    try:
        model = fetch_model(**model_params)
    except:
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
                model_name=model_name,
                epochs_interval=save_every_n_epochs
            )
        ]

        logging.info(f"Training the model for {epochs} epochs, saving every {save_every_n_epochs}")

        model.fit(
            x=train_set,
            validation_data=validation_set,
            batch_size=16,
            verbose=1,
            validation_split=validation_set,
            callbacks=callbacks,
            epochs=epochs
        )
