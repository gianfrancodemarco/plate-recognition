import os

from src.models.metrics import iou
from tensorflow.keras.models import load_model

model = load_model(
    os.getenv("MODEL_PATH", "/models/plate_recognition/raw_images.h5"),
    custom_objects={"iou": iou}
)

def predict(image):
    return model.predict(image)