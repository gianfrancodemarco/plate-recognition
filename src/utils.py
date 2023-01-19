import os
from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent
SRC_PATH = os.path.join(ROOT_PATH, "src")
DATA_PATH = os.path.join(ROOT_PATH, "data")
TESTS_PATH = os.path.join(ROOT_PATH, "tests")