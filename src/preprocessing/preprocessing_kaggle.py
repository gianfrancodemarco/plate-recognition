import os
import glob
import cv2
from lxml import etree
import csv
import logging

# create logger
logger = logging.getLogger(__name__)

IMAGE_SIZE = 256
dir_path = os.path.dirname(os.path.realpath(__file__))


class KaggleDatasetPreprocessor:
    KAGGLE_INPUT_PATH = os.path.join(dir_path, '..', '..', 'datasets', 'kaggle')
    KAGGLE_OUTPUT_PATH = os.path.join(dir_path, '..', '..', 'resources', 'kaggle')
    KAGGLE_INPUT_IMAGES = os.path.join(KAGGLE_INPUT_PATH, 'images')
    KAGGLE_INPUT_ANNOTATIONS = os.path.join(KAGGLE_INPUT_PATH, 'annotations')
    KAGGLE_OUTPUT_IMAGES = os.path.join(KAGGLE_OUTPUT_PATH, 'images')
    KAGGLE_OUTPUT_ANNOTATIONS = KAGGLE_OUTPUT_PATH

    def run_preprocessing(self):

        logger.info("Running preprocessing")

        if not os.path.exists(self.KAGGLE_OUTPUT_PATH):
            os.makedirs(self.KAGGLE_OUTPUT_PATH)
            logger.info("Created Kaggle folder")

        self.__resize_images()
        self.__annotations_to_csv()

        logger.info("Done preprocessing")

    def __resize_images(self):
        logger.info('Resizing images')

        if not os.path.exists(self.KAGGLE_OUTPUT_IMAGES):
            os.makedirs(self.KAGGLE_OUTPUT_IMAGES)
            logger.info("Created images folder")

        if len(os.listdir(self.KAGGLE_OUTPUT_IMAGES)):
            logger.info('Output folder is not empty, skipping')
            return

        data_path = os.path.join(self.KAGGLE_INPUT_IMAGES, '*g')
        files = glob.glob(data_path)
        files.sort()  # We sort the images in alphabetical order to match them to the xml files containing the annotations of the bounding boxes

        for f1 in files:
            img = cv2.imread(f1)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            cv2.imwrite(os.path.join(self.KAGGLE_OUTPUT_IMAGES, f1.split(os.sep)[-1]), img)
        logger.info('Done resizing')

    def __annotations_to_csv(self):
        logger.info('Resizing kaggle annotations and writing them to csv')

        annotations_file = os.path.join(self.KAGGLE_OUTPUT_ANNOTATIONS, 'annotations.csv')

        try:
            open(annotations_file)
            logger.info('Annotations file exists, skipping')
            return
        except:
            pass

        header = ['name', 'xmin', 'ymin', 'xmax', 'ymax']

        with open(annotations_file, 'w', newline='', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(header)

            text_files = [os.path.join(self.KAGGLE_INPUT_ANNOTATIONS, f) for f in
                          sorted(os.listdir(self.KAGGLE_INPUT_ANNOTATIONS))]
            for annotation_file in text_files:
                row = self.__resize_annotation(annotation_file)
                row = [annotation_file.split(os.sep)[-1]] + row
                writer.writerow(row)

        logger.info('Written annotations')

    def __resize_annotation(self, f):
        tree = etree.parse(f)
        for dim in tree.xpath("size"):
            width = int(dim.xpath("width")[0].text)
            height = int(dim.xpath("height")[0].text)
        for dim in tree.xpath("object/bndbox"):
            xmin = int(dim.xpath("xmin")[0].text) / (width / IMAGE_SIZE)
            ymin = int(dim.xpath("ymin")[0].text) / (height / IMAGE_SIZE)
            xmax = int(dim.xpath("xmax")[0].text) / (width / IMAGE_SIZE)
            ymax = int(dim.xpath("ymax")[0].text) / (height / IMAGE_SIZE)
        return [int(xmax), int(ymax), int(xmin), int(ymin)]
