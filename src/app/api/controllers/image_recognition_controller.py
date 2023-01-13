import io
import logging

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import ValidationError
from src.app.api.dtos.predict_image_dto import PredictImageDTO
from src.app.api.services.image_recognition_service import \
    ImageRecognitionService
from starlette.responses import StreamingResponse

router = APIRouter()


image_recognition_service = ImageRecognitionService()

@router.post("/predict/plate-bbox")
def predict_bbox(image_file: UploadFile):
    try:
        predict_image_dto = PredictImageDTO(image_file = image_file)
        numpy_image = np.asarray(predict_image_dto.image_file)
        bbox = image_recognition_service.predict_bbox(numpy_image)
        logging.warning(bbox)
        return {
            "data": {
                "bbox": bbox.tolist()
            }
        }
    except ValidationError as exc:
        logging.exception(exc)
        raise HTTPException(status_code=406, detail=str(exc.raw_errors[0].e)) from exc


@router.post("/predict/plate-bbox/annotate-image")
def predict_annotated_image(image_file: UploadFile):
    try:
        predict_image_dto = PredictImageDTO(image_file = image_file)
        numpy_image = np.asarray(predict_image_dto.image_file)
        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
        image = image_recognition_service.predict_bbox_and_annotate_image(numpy_image)
        res, im_png = cv2.imencode(".png", image)
        return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")
    except ValidationError as exc:
        logging.exception(exc)
        raise HTTPException(status_code=406, detail=str(exc.raw_errors[0].e)) from exc


@router.post("/predict/plate-text")
def predict_annotated_image(image_file: UploadFile):
    try:
        predict_image_dto = PredictImageDTO(image_file = image_file)
        numpy_image = np.asarray(predict_image_dto.image_file)
        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
        plate = image_recognition_service.predict_plate(numpy_image)
        return {
            "data": {
                "plate": plate
            }
        }
    except ValidationError as exc:
        logging.exception(exc)
        raise HTTPException(status_code=406, detail=str(exc.raw_errors[0].e)) from exc
