import json
import logging

import mlflow
from keras.models import Model
from src.models.train_callbacks import (EarlyStoppingByLossVal,
                                        OptunaPruneCallback,
                                        SaveModelMLFlowCallback)


class ModelTrainer:
    pass  # TODO


class MLFlowModelTrainer:

    def __init__(
        self,
        model_name: str,
        train_data,
        validation_data,
        batch_size: int = 16,
        epochs: int = 1000,
        verbose = 1,
        trial = None
    ) -> None:

        self.model_name = model_name
        self.train_data = train_data
        self.validation_data = validation_data
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.trial = trial

        self.__init_callbacks()

    def __init_callbacks(self):

        if self.trial:
            self.callbacks = [OptunaPruneCallback(trial=self.trial)]
        else:
            self.callbacks = [
                EarlyStoppingByLossVal(monitor='loss', value=1, verbose=1),
                SaveModelMLFlowCallback(
                    model_name=self.model_name
                )
            ]

    def train(
        self,
        model: Model,
        params_to_log: dict = None,
        results_path: str = None
    ):

        fit_func = lambda: self.__fit(
            model,
            results_path
        )

        if self.trial:
            fit_func()
        else:
            self.__mlflow_wrapped_fit(
                fit_func,
                params_to_log
            )


    def __fit(
        self,
        model: Model,
        results_path: str = None
    ):

        model.fit(
            epochs=self.epochs,
            x=self.train_data,
            validation_data=self.validation_data,
            batch_size=self.batch_size,
            verbose=self.verbose,
            callbacks=self.callbacks
        )

        if results_path:
            self.export_results(model, results_path)



    def __mlflow_wrapped_fit(
        self,
        fit_func: callable,
        params_to_log: dict = None
    ):

        with mlflow.start_run():

            if params_to_log:
                mlflow.log_params(params_to_log)

            mlflow.tensorflow.autolog(
                log_input_examples=True,
                log_models=True
            )

            logging.info(f"Training the model for {self.epochs} epochs")
            fit_func()

    def export_results(self, model: Model, results_path: str):
        with open(results_path, "w") as f:
            f.write(json.dumps(model.history.history, indent=4))
