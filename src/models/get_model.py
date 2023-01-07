import logging

import mlflow
from keras.layers import (Conv2D, Dense, Dropout, Flatten, Input, LeakyReLU,
                          MaxPooling2D)
from keras.metrics import RootMeanSquaredError
from keras.models import Sequential
from src.models.metrics import iou


def get_model(
    model_name: str = None,
    model_version: int = None,
    dropout: float = 0
):

    if not (model_name and model_version):
        raise ValueError("You must provide a name and a version for the model")

    try:
        model_version_uri = f"models:/{model_name}/{model_version}"
        model = mlflow.tensorflow.load_model(model_version_uri)
        logging.info(f"Loading registered model version from URI: '{model_version_uri}'")

    except Exception as exc:
        logging.exception(exc)

        input_shape = (256, 256, 3)
        model = Sequential()

        model.add(Input(shape=input_shape))

        # Only for 3 channel images
        model.add(Conv2D(32, 2))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Conv2D(32, 2))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Conv2D(32, 2))
        model.add(LeakyReLU(alpha=0.01))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, 2))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Conv2D(32, 2))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Conv2D(32, 2))
        model.add(LeakyReLU(alpha=0.01))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, 2))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Conv2D(32, 2))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Conv2D(32, 2))
        model.add(LeakyReLU(alpha=0.01))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, 2))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Conv2D(32, 2))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Conv2D(32, 2))
        model.add(LeakyReLU(alpha=0.01))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        if dropout:
            model.add(Dropout(dropout))

        # Only for 3 channel images
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.01))

        model.add(Dense(4))
        model.add(LeakyReLU(alpha=0.01))

        model.build()
        model.summary()

        model.compile(loss='mse', optimizer='adam', metrics=[RootMeanSquaredError(), iou])

    return model
