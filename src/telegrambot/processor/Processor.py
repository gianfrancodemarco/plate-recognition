import logging
import os
import cv2
import re

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
        #normal_images_model = load_model(os.path.join('processor', 'models', 'plate_detection_hitl_NormalImages.h5'))
        #bw_images_model = load_model(os.path.join('processor', 'models', 'plate_detection_hitl_BWImages.h5'))
        edge_detection_images_model = load_model(os.path.join('processor', 'models', 'plate_detection_hitl_LaplacianImages.h5'))
        normal_images_model_new = load_model(os.path.join('processor', 'models', 'plate_detection_hitl_NormalImages_1000.h5'))

        self.__plate_detection_models = [
            #(normal_images_model, self.get_normal_image),
            #(bw_images_model, self.get_bw_image),
            (edge_detection_images_model, self.get_edge_detection_image),
            (normal_images_model_new, self.get_normal_image)
        ]

        # self.__transformer_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
        # self.__transformer_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")
        self.__transformer_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
        self.__transformer_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")

    def get_normal_image(self, img):
        return img

    def get_bw_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    def get_edge_detection_image(self, img):
        # converting to gray scale
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # remove noise
        img = cv2.GaussianBlur(img, (11, 11), 1)

        # convolute with proper kernels
        img = cv2.Laplacian(img, cv2.CV_64F, ksize=3)

        return img

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
        # logger.info(image)

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

        images = []
        plate_bbox_predictions = []

        for (model, image_converter) in self.__plate_detection_models:
            logger.info("Performing image converting...")
            img = image_converter(np.copy(image))
            logger.info("Converted image!")
            logger.info(f"Shape: {img.shape}")

            plate_bbox_prediction = model.predict(np.array([img]))[0]
            img = cv2.rectangle(
                img,
                (int(plate_bbox_prediction[0]), int(plate_bbox_prediction[1])),
                (int(plate_bbox_prediction[2]), int(plate_bbox_prediction[3])),
                (255, 255, 255)
            )

            images.append(img)
            plate_bbox_predictions.append(plate_bbox_prediction)

        return images, plate_bbox_predictions

    def post_process_plates(self, plates):

        def clean_prediction(plate):
            return re.sub(r'\W+', '', plate)

        def discard_plate_too_long(plate):
            return not len(plate) > 10

        def discard_plate_too_short(plate):
            return not len(plate) < 4

        def get_plates_with_most_votest(plates):

            if len(plates) == 0:
                return []

            votes = {}
            for plate in plates:
                if plate in votes:
                    votes[plate] += 1
                else:
                    votes[plate] = 1

            list_of_tuple = [(k, v) for k, v in votes.items()]
            ordered_votes = sorted(list_of_tuple, key=lambda x: x[1], reverse=1)
            plates_to_keep = [ordered_votes[0]]  # always take the first
            for (plate, votes) in ordered_votes[1:]:
                if votes == plates_to_keep[0][1]:
                    plates_to_keep.append((plate, votes))

            plates_to_keep = [plate for (plate, votes) in plates_to_keep]
            return plates_to_keep

        plates = list(map(clean_prediction, plates))
        plates = list(filter(discard_plate_too_long, plates))
        plates = list(filter(discard_plate_too_short, plates))

        logging.info("PLATES HERE")
        plates = get_plates_with_most_votest(plates)
        logging.info("PLATES TO KEEP")

        return plates
