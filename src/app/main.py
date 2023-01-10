import logging
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from src import utils
from src.app.api import api
from starlette.middleware.cors import CORSMiddleware

if load_dotenv(
    dotenv_path=os.path.join(utils.ROOT_PATH, '.env'),
    override=True
):
    logging.info("Loaded environment variables")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "PUT", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=True,
)
app.include_router(api.api_router, prefix="/api/v1")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
