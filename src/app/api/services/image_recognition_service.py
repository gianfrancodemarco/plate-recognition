import asyncio
import logging
import os
from time import sleep

import cv2
import numpy as np
from keras.models import Model
from PIL import Image
from shapely.affinity import scale
from shapely.geometry import box
from src.data.image_preprocessing import crop_image, preprocess_image
from src.models.fetch_model import fetch_model
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

model_name = os.getenv("MODEL_NAME")
model_version = os.getenv("MODEL_VERSION")
tr_ocr_processor = os.getenv("TR_OCR_PROCESSOR", "microsoft/trocr-small-printed")
tr_ocr_model = os.getenv("TR_OCR_MODEL", "microsoft/trocr-small-printed")


class ImageRecognitionService:

    __detection_model: Model = None
    __transformer_processor: TrOCRProcessor = None
    __transformer_model: VisionEncoderDecoderModel = None

    def __init__(self) -> None:
        self.__load_models()

    def __load_detection_model(self) -> None:
        self.__detection_model = fetch_model(
            model_name=model_name,
            model_version=model_version
        )

    def __load_transformer_processor(self) -> None:
        self.__transformer_processor = TrOCRProcessor.from_pretrained(tr_ocr_processor)

    def __load_transformer_model(self) -> None:
        self.__transformer_model = VisionEncoderDecoderModel.from_pretrained(tr_ocr_model)

    def __load_models(self):
        logging.info("Downloading models in background")
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, self.__load_detection_model)
        loop.run_in_executor(None, self.__load_transformer_processor)
        loop.run_in_executor(None, self.__load_transformer_model)


    def __get_detection_model(self):
        if not self.__detection_model:
            self.__load_detection_model()
        return self.__detection_model

    def __get_transformer_processor(self):
        if not self.__transformer_processor:
           self.__load_transformer_processor()
        return self.__transformer_processor

    def __get_transformer_model(self):
        if not self.__transformer_model:
            self.__load_transformer_model()
        return self.__transformer_model

    def __predict_image_bbox__(self, image: np.ndarray):
        _image = image.copy()
        _image = preprocess_image(_image)
        _image_batch = np.array([_image])
        return self.__get_detection_model().predict(_image_batch)[0]

    def predict_bbox(self, image: np.ndarray) -> np.ndarray:
        return self.__predict_image_bbox__(image)

    def predict_bbox_and_annotate_image(self, image: np.ndarray):
        bbox = self.__predict_image_bbox__(image)
        bbox = box(*bbox)

        yfact = image.shape[0]/255
        xfact = image.shape[1]/255

        bbox = scale(bbox, xfact=xfact, yfact=yfact, origin=(0, 0))

        pt1 = (int(bbox.bounds[0]), int(bbox.bounds[1]))
        pt2 = (int(bbox.bounds[2]), int(bbox.bounds[3]))

        _image = image.copy()
        cv2.rectangle(_image, pt1, pt2, color=(0, 0, 255), thickness=3)
        return _image

    def predict_plate(self, image: np.ndarray):

        bbox = self.__predict_image_bbox__(image)

        cropped_image = crop_image(image, bbox)
        cropped_image = Image.fromarray(cropped_image)

        transformer_processor = self.__get_transformer_processor()
        transformer_model = self.__get_transformer_model()
        pixel_values = transformer_processor(cropped_image, return_tensors="pt").pixel_values
        generated_ids = transformer_model.generate(pixel_values)
        generated_text = transformer_processor.batch_decode(
            generated_ids, skip_special_tokens=True)[0]

        return generated_text
