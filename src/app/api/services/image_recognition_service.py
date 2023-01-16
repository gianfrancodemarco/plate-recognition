import asyncio
import logging
import os
from time import sleep

import cv2
import numpy as np
from keras.models import Model, load_model
from PIL import Image
from shapely.affinity import scale
from shapely.geometry import box
from src.models.metrics import iou
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class ImageRecognitionService:

    __detection_model: Model = None
    __transformer_processor: TrOCRProcessor = None
    __transformer_model: VisionEncoderDecoderModel = None

    def __init__(self) -> None:

        def __load_detection_model():
            try:
                return load_model(
                        os.getenv("MODEL_PATH"),
                        custom_objects={"iou": iou}
                    )
            except:
                return __load_detection_model()

        def __load_transformer_processor():
            try:
                return TrOCRProcessor.from_pretrained(
                    "microsoft/trocr-small-printed")
            except:
                return __load_transformer_processor()

        def __load_transformer_model():
            try:
                return VisionEncoderDecoderModel.from_pretrained(
                    "microsoft/trocr-small-printed")
            except:
                return __load_transformer_model()

        def _load():
            logging.info("Downloading models in background")
            self.__detection_model = __load_detection_model()
            self.__transformer_processor = __load_transformer_processor()
            self.__transformer_model = __load_transformer_model()

        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _load)

    def __get_detection_model(self):
        while not self.__detection_model:
            logging.warning("__detection_model not ready. Sleeping 1 sec.")
            sleep(1)
        return self.__detection_model

    def __get_transformer_processor(self):
        while not self.__transformer_processor:
            logging.warning("__transformer_processor not ready. Sleeping 1 sec.")
            sleep(1)
        return self.__transformer_processor

    def __get_transformer_model(self):
        while not self.__transformer_model:
            logging.warning("__transformer_model not ready. Sleeping 1 sec.")
            sleep(1)
        return self.__transformer_model

    def __predict_image_bbox__(self, image: np.ndarray):
        _image = image.copy()
        _image = self.preprocess_image(_image)
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

        cropped_image = self.crop_image(image, bbox)
        cropped_image = Image.fromarray(cropped_image)

        transformer_processor = self.__get_transformer_processor()
        transformer_model = self.__get_transformer_model()
        pixel_values = transformer_processor(cropped_image, return_tensors="pt").pixel_values
        generated_ids = transformer_model.generate(pixel_values)
        generated_text = transformer_processor.batch_decode(
            generated_ids, skip_special_tokens=True)[0]

        return generated_text

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        image = cv2.resize(image, (256, 256))
        return image

    def crop_image(self, image, bbox: box):

        _cropped_image = image.copy()
        _cropped_image = _cropped_image[int(bbox[3]):int(bbox[1]), int(bbox[2]):int(bbox[0])]
        return _cropped_image
