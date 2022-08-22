import logging
import os
import cv2

import numpy as np
from keras.models import load_model
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)



class Processor:

    def __init__(self):
        self.__plate_detection_model = load_model(os.path.join('processor', 'models', 'plate_detection_model.h5'))
        #self.__transformer_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
        #self.__transformer_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")
        self.__transformer_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
        self.__transformer_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")

    def resize_image(self, image):
        return cv2.resize(image, (256, 256))

    def edge_detection(self, image):
        # converting to gray scale
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # remove noise
        image = cv2.GaussianBlur(image, (11, 11), 1)

        # convolute with proper kernels
        image = cv2.Laplacian(image, cv2.CV_64F, ksize=3)

        return image

    def crop_plate(self, image, predictions):
        cropped_image = image[int(predictions[3]): int(predictions[1]), int(predictions[2]): int(predictions[0])]
        return cropped_image

    def get_ocr_prediction(self, image, predictions):

        image = self.crop_plate(image, predictions)
        logger.info(image)

        cv2.imwrite('img.png', image)
        image = Image.open('img.png').convert("RGB")

        pixel_values = self.__transformer_processor(image, return_tensors="pt").pixel_values
        generated_ids = self.__transformer_model.generate(pixel_values)
        generated_text = self.__transformer_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return generated_text

    def get_plate_detection_prediction(self, image):

        logger.info("Resizing the photo...")
        image = self.resize_image(image)
        logger.info("Photo resized!")
        logger.info(f"Shape: {image.shape}")

        logger.info("Performing edge detection...")
        image = self.edge_detection(image)
        logger.info("Edge detection done!")
        logger.info(f"Shape: {image.shape}")

        plate_bbox_prediction = self.__plate_detection_model.predict(np.array([image]))[0]
        image = cv2.rectangle(
            image,
            (int(plate_bbox_prediction[0]), int(plate_bbox_prediction[1])),
            (int(plate_bbox_prediction[2]), int(plate_bbox_prediction[3])),
            (255, 255, 255)
        )
        return image, plate_bbox_prediction

