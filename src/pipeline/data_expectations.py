import json
import logging
import os

import great_expectations as ge
import pandas as pd
from src import utils

PROCESSED_PATH = os.path.join(utils.DATA_PATH, "processed")
GREAT_EXPECTATIONS_REPORTS_PATH = os.path.join(utils.REPORTS_PATH, "great_expectations")

for split in ["train", "validation", "test"]:
    
    annotations_file = os.path.join(PROCESSED_PATH, split, "annotations.csv")
    annotations = pd.read_csv(annotations_file)
    df = ge.dataset.PandasDataset(annotations)

    result = df.expect_table_columns_to_match_ordered_list(["name","minx","miny","maxx","maxy","plate"])
    
    output_path = os.path.join(GREAT_EXPECTATIONS_REPORTS_PATH, f"{split}_annotations.json")

    with open(output_path, "w") as f:
        f.write(json.dumps(result.to_json_dict(), indent=4))

    if not result["success"]:
        logging.warning(f"Expectation on data columns where not matched. Check the output at {output_path}")
        raise Exception()
