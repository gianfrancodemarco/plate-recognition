import logging

import mlflow
from keras.callbacks import Callback


class SaveModelMLFlowCallback(Callback):

    def __init__(
        self,
        model_name: str,
        epochs_interval: int
    ):
        super().__init__()
        self.model_name = model_name
        self.epochs_interval = epochs_interval

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 0 and epoch % self.epochs_interval == 0:
            # https://mlflow.org/docs/latest/model-registry.html#adding-an-mlflow-model-to-the-model-registry
            # Replace with method 1
            run_id = mlflow.active_run().info.run_id
            artifact_path = "model"
            model_uri = f"runs:/{run_id}/{artifact_path}"
            model_details = mlflow.register_model(model_uri=model_uri, name=self.model_name)
            logging.info(f"Saved model at {model_details}")


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


def train_model(
        model,
        dataset,
        epochs: int = 1000,
        validation_split: float = 0,
        validation_dataset = None,
        early_stopping: bool = False,
        save_every_n_epochs: int = 3,
        model_name: None = "model",
        batch_size: int = 16
):
    callbacks = []
    if early_stopping:
        callbacks.append(EarlyStoppingByLossVal(monitor='loss', value=1, verbose=1))

    if save_every_n_epochs:
        callbacks.append(SaveModelMLFlowCallback(
            model_name=model_name,
            epochs_interval=save_every_n_epochs
        ))

    logging.info(f"Training the model for {save_every_n_epochs} epochs")
    model.fit(
        x=dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_split=validation_split,
        callbacks=callbacks
    )
    logging.info(f"Model fitted for {save_every_n_epochs} epochs")

    return model
