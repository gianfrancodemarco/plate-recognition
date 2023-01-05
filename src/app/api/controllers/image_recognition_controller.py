import logging

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import ValidationError
from src.app.api.services import image_recognition_service
from src.app.api.validators.image_validator import ImageValidator

router = APIRouter()

@router.post("/predict")
def predict(image_file: UploadFile):
    try:
        ImageValidator(image=image_file)
        return image_recognition_service.predict(image_file)
    except ValidationError as e:
        logging.exception(e)
        raise HTTPException(status_code=406, detail=str(e.raw_errors[0].e)) from e
