import logging
import math
import os
import shutil

import pandas as pd
from sklearn.utils import shuffle
from src import utils

RAW_IMAGES_PATH = os.path.join(utils.DATA_PATH, "raw", "images")
PROCESSED_PATH = os.path.join(utils.DATA_PATH, "processed")

ANNOTATIONS_PATH = os.path.join(utils.DATA_PATH, "raw", "annotations.csv")
PLATES_PATH = os.path.join(utils.DATA_PATH, "raw", "plates.csv")
TRAIN_SET_FRACTION = os.getenv("TRAIN_SET_FRACTION", 0.7)
TEST_SET_FRACTION = os.getenv("TEST_SET_FRACTION", 0.2)
VALIDATION_SET_FRACTION = os.getenv("VALIDATION_SET_FRACTION", 0.1)


import tensorflow as tf
import numpy as np
import random 
tf.random.set_seed(42)
np.random.seed(42)
random.seed(10)


def merge_annotations_and_plates_dataframes():
    # Read annotations (bboxes) and plates to join into a single dataframe
    annotations = pd.read_csv(ANNOTATIONS_PATH)
    plates = pd.read_csv(PLATES_PATH)
    annotations["name"] = annotations["image_name"]
    annotations = annotations.set_index("name").join(plates.set_index("name"))
    annotations = annotations.reset_index()
    return annotations

def reorganize_annotations_columns(annotations):
    annotations["minx"] = annotations["bbox_x"]
    annotations["miny"] = annotations["bbox_y"]
    annotations["maxx"] = annotations["bbox_x"] + annotations["bbox_width"]
    annotations["maxy"] = annotations["bbox_y"] + annotations["bbox_height"]
    annotations = annotations[["name", "minx", "miny", "maxx", "maxy", "plate"]].copy()
    return annotations

def make_dataset():
    annotations = merge_annotations_and_plates_dataframes()
    annotations = reorganize_annotations_columns(annotations)
    annotations = shuffle(annotations)

    # Split the dataset
    n_data = len(annotations)
    current_index = 0

    for (split, fraction) in [("train", TRAIN_SET_FRACTION), ("test", TEST_SET_FRACTION), ("validation", VALIDATION_SET_FRACTION)]:

        # Take the next n_data_in_split from the dataset
        n_data_in_split = math.ceil(n_data * fraction)
        data_in_split = annotations.iloc[current_index: current_index + n_data_in_split]
        current_index += n_data_in_split

        # Copy annotations into split's path
        split_path = os.path.join(PROCESSED_PATH, split)
        if not os.path.exists(split_path):
            os.makedirs(split_path)

        data_in_split.to_csv(os.path.join(split_path, "annotations.csv"), index=False)

        # Copy images into split's path
        split_images_path = os.path.join(PROCESSED_PATH, split, "images")
        if not os.path.exists(split_images_path):
            os.makedirs(split_images_path)

        for sample in data_in_split.values.tolist():
            image_name = str(sample[0])
            image_source_path = os.path.join(RAW_IMAGES_PATH, image_name)
            image_dest_path = os.path.join(split_images_path, image_name)
            shutil.copyfile(image_source_path, image_dest_path)


if __name__ == "__main__":
    make_dataset()
