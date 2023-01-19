import logging
import logging.config
import os

from dotenv import load_dotenv

from src import utils

logging.config.fileConfig(
    os.path.join(utils.SRC_PATH, "logging.conf"),
    disable_existing_loggers=False
)

if load_dotenv(
    dotenv_path=os.path.join(utils.ROOT_PATH, '.env'),
    override=True
):
    logging.info("Loaded environment variables")


import tensorflow as tf
import numpy as np
import random 
tf.random.set_seed(42)
np.random.seed(None)
random.seed(10)