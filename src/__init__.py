# Configure logging
import logging.config
import os

from src import utils

logging.config.fileConfig(
    os.path.join(utils.SRC_PATH, "logging.conf"),
    disable_existing_loggers=False
)
