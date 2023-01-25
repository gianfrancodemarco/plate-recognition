import os

import tensorflow as tf
from keras.layers import Dense, Flatten, Input
from src import utils
from src.features.dataset import get_dataset
from src.models.mlflow_model_trainer import MLFlowModelTrainer
from src.models.model_builder import ModelBuilder

os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.sqlite.db"

ANNOTATIONS_PATH = os.path.join(utils.TESTS_PATH, "assets", "annotations.csv")

class TestModel():
    def test_model_overfit(self):
        
        model = ModelBuilder().build()
        train = get_dataset(annotations_path=ANNOTATIONS_PATH)
        train = train.take(1)
        
        trainer = MLFlowModelTrainer(
            epochs=10,
            model_name="test",
            train_data=train,
        )

        trainer.train(model)

        loss = model.history