import os

from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, Input, Dropout, BatchNormalization, \
    LeakyReLU
from keras.callbacks import Callback
from keras.metrics import MeanSquaredError, RootMeanSquaredError
from src.main.metrics import iou

class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.0001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            print("\nEarly stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("\nEpoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


def build_raw_image_model(dropout: float = 0):
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


def build_bw_image_model(dropout: float = 0):
    input_shape = (256, 256, 1)
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


def build_edge_detection_image_model(dropout: bool = False):
    input_shape = (256, 256, 1)
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


def train_model(
        model,
        X,
        y,
        epochs=1000,
        validation_split: float = 0,
        early_stopping: bool = False,
        save_every_n_epochs: int = 100,
        save_path = None,
        model_name = None
):

    current_epochs = 0
    model_name += '.h5'

    callbacks = []
    if early_stopping:
        callbacks = [EarlyStoppingByLossVal(monitor='loss', value=1, verbose=1)]

    while current_epochs < epochs:
        current_epochs += save_every_n_epochs

        print(f"Training the model for {save_every_n_epochs} epochs")

        model.fit(
            X,
            y,
            epochs=save_every_n_epochs,
            batch_size=16,
            verbose=1,
            validation_split=validation_split,
            callbacks=callbacks
        )
        print("Model trained")

        if save_every_n_epochs:
            print("Saving the model")
            model.save(os.path.join(save_path, model_name))
            print(f"Saved model {model_name}")

    return model
