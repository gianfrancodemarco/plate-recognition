# Configure logging
import logging.config
import os
from dotenv import load_dotenv
from src import utils

if load_dotenv(
    dotenv_path=os.path.join(utils.ROOT_PATH, '.env'),
    override=True
):
    logging.info("Loaded environment variables")


logging.config.fileConfig(
    os.path.join(utils.SRC_PATH, "logging.conf"),
    disable_existing_loggers=False
)