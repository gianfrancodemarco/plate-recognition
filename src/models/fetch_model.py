import logging

import mlflow

def fetch_model(
    model_name: str = None,
    model_version: int = None,
):
    """
    Fetches the corresponding model from MLFlow artifacts storage
    If it doesn't exists, creates a new model
    """

    if not (model_name and model_version):
        raise ValueError("You must provide a name and a version for the model")

    model_version_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.tensorflow.load_model(
        model_version_uri,
        keras_model_kwargs={"custom_objects":{}}
    )
    logging.info(f"Loaded registered model version from URI: '{model_version_uri}'")
    return model
