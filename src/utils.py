import functools
import json
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


def dict2obj(__dict__: dict, object_hook: callable):
    """
    Converts a nested dict into an object accessible with dot notation
    
    ``__dict__`` is the object to convert
    """
    return json.loads(json.dumps(__dict__), object_hook=object_hook)


def rgetattr(obj, attr, *args):
    """
    Implements recursive getattr on objects to support dot notation
    """
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))
