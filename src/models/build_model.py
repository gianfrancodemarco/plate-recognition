import logging

from keras.layers import (Conv2D, Dense, Dropout, Flatten, Input, LeakyReLU,
                          MaxPooling2D, ReLU, BatchNormalization)
from keras.metrics import RootMeanSquaredError
from keras.models import Sequential
from src.models.metrics import iou
from keras.optimizers.optimizer_experimental.sgd import SGD
from keras.optimizers.optimizer_experimental.adam import Adam

class ModelBuilder:

    def __init__(
        self,
        dropout: float = 0,
        cnn_blocks: int = 1,
        filters_num: int = 32,
        filters_kernel_size: int = (2, 2),
        optimizer_name: str = "adam",
        learning_rate: float = 0.001        
    ) -> None:
        self.dropout = dropout
        self.cnn_blocks = cnn_blocks
        self.filters_num = filters_num
        self.filters_kernel_size = filters_kernel_size
        self.optimizer_name = optimizer_name 
        self.learning_rate = learning_rate

    def build(self):

        model = Sequential()

        input_shape = (256, 256, 3)
        model.add(Input(shape=input_shape))

        for i in range(self.cnn_blocks):
            model.add(Conv2D(self.filters_num, self.filters_kernel_size))
            model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        if self.dropout:
            model.add(Dropout(self.dropout))

        model.add(Dense(128))
        model.add(BatchNormalization())
        model.add(Dense(4))
        model.add(LeakyReLU(0.001))

        model.build()
        model.summary()

        optimizer = self.get_optimizer()
        model.compile(loss='mse', optimizer=optimizer, metrics=[RootMeanSquaredError()])

        return model

    def get_optimizer(self):
        if self.optimizer_name == "adam":
            return Adam(learning_rate=self.learning_rate)
        elif self.optimizer_name == "sgd":
            return SGD(learning_rate=self.learning_rate)