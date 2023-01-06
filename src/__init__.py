from src import utils

# Configure logging
import logging.config
import os

logging.config.fileConfig(
    os.path.join(utils.SRC_PATH, "logging.conf"),
    disable_existing_loggers=False
)