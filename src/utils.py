import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT_PATH = Path(__file__).parent.parent
SRC_PATH = os.path.join(ROOT_PATH, "src")
DATA_PATH = os.path.join(ROOT_PATH, "data")
TESTS_PATH = os.path.join(ROOT_PATH, "tests")

def set_random_states(random_state):

    if not random_state:
        random_state = 42

    tf.random.set_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)
