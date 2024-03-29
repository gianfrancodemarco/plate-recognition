import logging
import os

import dvc.api
import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback
from optuna.trial import Trial
from src import utils
from src.features.dataset import get_dataset
from src.features.dataset_generator import ImageDatasetType
from src.models.model_builder import ModelBuilder
from src.models.mlflow_model_trainer import MLFlowModelTrainer
from src.pipeline.param_parser import ParamParser

DATASETS_BASE = os.path.join(utils.DATA_PATH, "processed")

params_dict = dvc.api.params_show()
params = ParamParser().parse(params_dict)
utils.set_random_states(params.random_state)



def objective(params, trial: Trial):
    """Objective function for optimization trials."""
    # Parameters to tune
    params.train.model.cnn_blocks = trial.suggest_int("cnn_blocks", 0, 6)
    params.train.model.dropout = trial.suggest_float("dropout", 0, 0.9, step=0.1)
    params.train.model.filters_num = trial.suggest_int("filters", 16, 64, 8)
    optimizer_name=trial.suggest_categorical(name="optimizer_name", choices=["adam", "sgd"])
    learning_rate=trial.suggest_float("learning_rate", 0.0005, 0.1, step=0.001)
    augmentation=trial.suggest_int("augmentation", 0, 1)

    dataset_generator_type = ImageDatasetType.BBOX_IMAGES_DATASET_GENERATOR
    if augmentation:
        dataset_generator_type = ImageDatasetType.BBOX_AUGMENTED_IMAGES_DATASET_GENERATOR

    
    train_set = get_dataset(
        annotations_path=os.path.join(DATASETS_BASE, "train", "annotations.csv"),
        dataset_generator_type=dataset_generator_type
    )
    validation_set = get_dataset(
        annotations_path=os.path.join(DATASETS_BASE, "validation", "annotations.csv"),
        dataset_generator_type=dataset_generator_type
    )


    # Train & evaluate
    model = ModelBuilder(
        dropout=params.train.model.dropout,
        cnn_blocks=params.train.model.cnn_blocks,
        filters_num=params.train.model.filters_num,
        filters_kernel_size=params.train.model.filters_kernel_size,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate
    ).build()

    model_trainer = MLFlowModelTrainer(
        epochs=20,
        model_name=params.train.model.model_name,
        train_data=train_set,
        validation_data=validation_set,
        trial=trial
    )

    model_trainer.train(
        model,
        params_to_log=params.__dict__
    )

    trial.set_user_attr("loss", model.history.history["loss"][-1])
    trial.set_user_attr("val_loss", model.history.history["val_loss"][-1])

    val_loss = model.history.history["val_loss"][-1]
    return val_loss


if __name__ == "__main__":

    NUM_TRIALS = 500

    # Optimize
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(study_name="Optimization", direction="minimize", pruner=pruner)
    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name="val_loss"
    )
    study.optimize(
        lambda trial: objective(params, trial),
        n_trials=NUM_TRIALS,
        callbacks=[mlflow_callback]
    )
