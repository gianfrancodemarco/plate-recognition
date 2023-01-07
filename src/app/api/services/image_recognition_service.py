import os

import cv2
import numpy as np
from keras.models import Model, load_model
from shapely.affinity import scale
from shapely.geometry import box
from src.models.metrics import iou


class ImageRecognitionService:

    model: Model = None

    def __get_model__(self):
        if not self.model:
            self.model = load_model(
                os.getenv("MODEL_PATH"),
                custom_objects={"iou": iou}
            )
        return self.model

    def __predict_image_bbox__(self, image: np.ndarray):
        _image = image.copy()
        _image = self.preprocess_image(_image)
        _image_batch = np.array([_image])
        return self.__get_model__().predict(_image_batch)[0]

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

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        image = cv2.resize(image, (256, 256))
        return image
