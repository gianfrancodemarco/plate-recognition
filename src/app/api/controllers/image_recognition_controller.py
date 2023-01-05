from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import ValidationError
from src.app.api.validators.image_validator import ImageValidator

router = APIRouter()

@router.post("/predict")
def predict(image_file: UploadFile):
    try:
        ImageValidator(image_file)
        res = {"result": "ok"}
        return res
    except ValidationError as e:
        raise HTTPException(status_code=406, detail=str(e.raw_errors[0].e)) from e
