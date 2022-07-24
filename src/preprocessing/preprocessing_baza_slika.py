import shutil
import os
import cv2
import logging

logger = logging.getLogger(__name__)

IMAGE_SIZE = 256

class BazaSlikaDatasetPreprocessor:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    BAZA_SLIKA_PATH = f'{dir_path}/../../datasets/baza_slika'
    OUTPUT_PATH = f'{dir_path}/../../resources/baza_slika/images'

    def run_preprocessing(self):

        logger.info("Running preprocessing")

        self.__resize_images()

        logger.info("Done preprocessing")

    def __resize_images(self):

        logger.info('Resizing images')

        if not os.path.exists(self.OUTPUT_PATH):
            os.makedirs(self.OUTPUT_PATH)
            logger.info("Created images folder")

        if len(os.listdir(self.OUTPUT_PATH)):
            logger.info('Output folder is not empty, skipping')
            return

        idx = 0
        for f in os.scandir(self.BAZA_SLIKA_PATH):
            if f.is_dir():
                for image in os.listdir(f.path):

                    logger.debug(image)

                    img = cv2.imread(os.path.join(f.path, image))
                    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                    cv2.imwrite(os.path.join(self.OUTPUT_PATH, f"{idx}.jpg"), img)

                    idx += 1

        logger.info('Done resizing')
