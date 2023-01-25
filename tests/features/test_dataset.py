import os

import tensorflow as tf
from src import utils
from src.features.dataset import get_dataset

ANNOTATIONS_PATH = os.path.join(utils.TESTS_PATH, "assets", "annotations.csv")

class TestDataset:

    def test_get_dataset(self):
        dataset = get_dataset(annotations_path=ANNOTATIONS_PATH)

        assert isinstance(dataset, tf.data.Dataset)
        dataset.take(1)