import logging

import mlflow
from keras.layers import (Conv2D, Dense, Dropout, Flatten, Input, LeakyReLU, ReLU,
                          MaxPooling2D)
from keras.metrics import RootMeanSquaredError
from keras.models import Sequential
from src.models.metrics import iou

def get_model(
    model_name: str = None,
    model_version: int = None,
    dropout: float = 0,
    cnn_blocks: int = 1,
    filters_num: int = 32,
    filters_kernel_size: int = (2,2)
):
    """
    Fetches the corresponding model from MLFlow artifacts storage
    If it doesn't exists, creates a new model
    """

    if not (model_name and model_version):
        raise ValueError("You must provide a name and a version for the model")

    try:
        model_version_uri = f"models:/{model_name}/{model_version}"
        model = mlflow.tensorflow.load_model(
            model_version_uri, 
            keras_model_kwargs={"custom_objects":{"iou": iou}}
        )
        logging.info(f"Loaded registered model version from URI: '{model_version_uri}'")

    except Exception as exc:
        logging.exception(exc)

        model = Sequential(name=model_name)

        input_shape = (256, 256, 3)
        model.add(Input(shape=input_shape))

        for i in range(cnn_blocks):
            model.add(Conv2D(filters_num, (2, 2)))
            model.add(ReLU())
            model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        if dropout:
            model.add(Dropout(dropout))

        model.add(Dense(128))
        model.add(Dense(4))
        model.add(ReLU())

        model.build()
        model.summary()

        model.compile(loss='mse', optimizer='adam', metrics=[RootMeanSquaredError(), iou])

    return model
