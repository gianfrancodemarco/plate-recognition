import logging

import mlflow
from keras.callbacks import Callback

class SaveModelMLFlowCallback(Callback):

    def __init__(
        self,
        model_name: str,
    ):
        super().__init__()
        self.model_name = model_name

    def save_model(self):
        run_id = mlflow.active_run().info.run_id
        artifact_path = "model"
        model_uri = f"runs:/{run_id}/{artifact_path}"
        model_details = mlflow.register_model(model_uri=model_uri, name=self.model_name)
        return model_details

    def on_train_end(self, epoch, logs=None):
        model_details = self.save_model()
        logging.info(f"Saved final model at {model_details}")


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
