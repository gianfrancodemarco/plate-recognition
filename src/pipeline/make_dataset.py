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


def make_dataset():

    # Read annotations (bboxes) and plates to join into a single dataframe
    annotations = pd.read_csv(ANNOTATIONS_PATH)
    plates = pd.read_csv(PLATES_PATH)
    annotations = annotations.set_index("name").join(plates.set_index("name"))
    annotations = annotations.reset_index()
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
    if len(os.listdir(PROCESSED_PATH)) == 0:
        make_dataset()
    else:
        logging.warning(
            f"{PROCESSED_PATH} is not empty. Delete its content and try again if you want to generate a new path")
