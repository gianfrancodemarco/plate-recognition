import os

import pytest
import tensorflow as tf
from keras.layers import Dense, Flatten, Input
from src import utils
from src.features.dataset import get_dataset
from src.models.mlflow_model_trainer import MLFlowModelTrainer

os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.sqlite.db"

ANNOTATIONS_PATH = os.path.join(utils.TESTS_PATH, "assets", "annotations.csv")

class TestMlflowModelTrainer():
    def test_mlflow_model_trainer_error(self):
        with pytest.raises(TypeError):
            MLFlowModelTrainer()

    def test_mlflow_model_trainer(self):

        train = get_dataset(annotations_path=ANNOTATIONS_PATH)
        validation = get_dataset(annotations_path=ANNOTATIONS_PATH)

        model = tf.keras.Sequential([
            Input(shape=(256,256,3)),
            Flatten(),
            Dense(2, activation='relu'),
            Dense(4)
        ])
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['accuracy']
        )

        trainer = MLFlowModelTrainer(
            model_name="test",
            train_data=train,
            validation_data=validation
        )

        trainer.train(model)
