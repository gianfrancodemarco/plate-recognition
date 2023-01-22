import os

import great_expectations as ge
import pandas as pd
from src import utils
import logging
PROCESSED_PATH = os.path.join(utils.DATA_PATH, "processed")

ANNOTATION_FILES = [
    os.path.join(PROCESSED_PATH, "train", "annotations.csv"),
    os.path.join(PROCESSED_PATH, "validation", "annotations.csv"),
    os.path.join(PROCESSED_PATH, "test", "annotations.csv")   
]

for annotation_file in ANNOTATION_FILES:
    annotations = pd.read_csv(annotation_file)
    df = ge.dataset.PandasDataset(annotations)
    result = df.expect_table_columns_to_match_ordered_list(["name","minx","miny","maxx","maxy","plate"])

    if not result["success"]:
        logging.warning("Expectation on data columns where not matched")
        logging.info(result)
        raise Exception()