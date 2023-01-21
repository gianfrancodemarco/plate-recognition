import logging

from keras.layers import (Conv2D, Dense, Dropout, Flatten, Input, LeakyReLU,
                          MaxPooling2D, ReLU, BatchNormalization)
from keras.metrics import RootMeanSquaredError
from keras.models import Sequential
from src.models.metrics import iou


def build_model(
    dropout: float = 0,
    cnn_blocks: int = 1,
    filters_num: int = 32,
    filters_kernel_size: int = (2, 2),
    **kwargs
):

    model = Sequential()

    input_shape = (256, 256, 3)
    model.add(Input(shape=input_shape))

    for i in range(cnn_blocks):
        model.add(Conv2D(filters_num, filters_kernel_size))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    if dropout:
        model.add(Dropout(dropout))

    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Dense(4))
    model.add(LeakyReLU(0.001))

    model.build()
    model.summary()

    model.compile(loss='mse', optimizer='adam', metrics=[RootMeanSquaredError()])

    return model
