from fastapi import APIRouter
from src.app.api.controllers import image_recognition_controller

api_router = APIRouter()
api_router.include_router(image_recognition_controller.router, prefix="/image-recognition")
