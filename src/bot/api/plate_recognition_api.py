import os
from io import BytesIO

import requests

PLATE_RECOGNITION_APP_SCHEMA = os.getenv("PLATE_RECOGNITION_APP_SCHEMA", "http")
PLATE_RECOGNITION_APP_HOST = os.getenv("PLATE_RECOGNITION_APP_HOST", "plate-recognition-app")
PLATE_RECOGNITION_APP_PORT = os.getenv("PLATE_RECOGNITION_APP_PORT", "8080")
GET_PLATE_TEXT_URL = f"{PLATE_RECOGNITION_APP_SCHEMA}://{PLATE_RECOGNITION_APP_HOST}:{PLATE_RECOGNITION_APP_PORT}/api/v1/image-recognition/predict/plate-text?postprocess=true"


def get_plate_text(photo: BytesIO):
    return requests.post(
        GET_PLATE_TEXT_URL,
        files={
            "image_file": photo
        }
    )
