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


class LicensePlateRecognitionDatasetPreprocessor:

    LICENSE_PLATE_RECOGNITION_INPUT_PATH = os.path.join(dir_path, '..', '..', 'datasets', 'license-plate-recognition')
    LICENSE_PLATE_RECOGNITION_OUTPUT_PATH = os.path.join(dir_path, '..', '..', 'resources', 'license_plate_recognition')
    LICENSE_PLATE_RECOGNITION_INPUT_IMAGES = os.path.join(LICENSE_PLATE_RECOGNITION_INPUT_PATH, 'images')
    LICENSE_PLATE_RECOGNITION_INPUT_ANNOTATIONS = os.path.join(LICENSE_PLATE_RECOGNITION_INPUT_PATH, 'annotations')
    LICENSE_PLATE_RECOGNITION_OUTPUT_IMAGES = os.path.join(LICENSE_PLATE_RECOGNITION_OUTPUT_PATH, 'images')
    LICENSE_PLATE_RECOGNITION_OUTPUT_ANNOTATIONS = LICENSE_PLATE_RECOGNITION_OUTPUT_PATH

    def run_preprocessing(self):

        logger.info("Running preprocessing")

        if not os.path.exists(self.LICENSE_PLATE_RECOGNITION_OUTPUT_PATH):
            os.makedirs(self.LICENSE_PLATE_RECOGNITION_OUTPUT_PATH)
            logger.info("Created License plate recognition folder")

        self.__resize_images()
        self.__annotations_to_csv()

        logger.info("Done preprocessing")

    def __resize_images(self):
        logger.info('Resizing images')

        if not os.path.exists(self.LICENSE_PLATE_RECOGNITION_OUTPUT_IMAGES):
            os.makedirs(self.LICENSE_PLATE_RECOGNITION_OUTPUT_IMAGES)
            logger.info("Created images folder")

        if len(os.listdir(self.LICENSE_PLATE_RECOGNITION_OUTPUT_IMAGES)):
            logger.info('Output folder is not empty, skipping')
            return

        data_path = os.path.join(self.LICENSE_PLATE_RECOGNITION_INPUT_IMAGES, '*g')
        files = glob.glob(data_path)
        files.sort()  # We sort the images in alphabetical order to match them to the xml files containing the annotations of the bounding boxes

        for idx, f1 in enumerate(files):
            img = cv2.imread(f1)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            #cv2.imwrite(os.path.join(self.LICENSE_PLATE_RECOGNITION_OUTPUT_IMAGES, f1.split(os.sep)[-1]), img)
            cv2.imwrite(os.path.join(self.LICENSE_PLATE_RECOGNITION_OUTPUT_IMAGES, str(idx) + '.jpg'), img)

        logger.info('Done resizing')

    def __annotations_to_csv(self):
        logger.info('Resizing license plate recognition annotations and writing them to csv')

        annotations_file = os.path.join(self.LICENSE_PLATE_RECOGNITION_OUTPUT_ANNOTATIONS, 'annotations.csv')

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

            text_files = [os.path.join(self.LICENSE_PLATE_RECOGNITION_INPUT_ANNOTATIONS, f) for f in
                          sorted(os.listdir(self.LICENSE_PLATE_RECOGNITION_INPUT_ANNOTATIONS))]
            for idx, annotation_file in enumerate(text_files):
                row = self.__resize_annotation(annotation_file)

                row = [str(idx)] + row
                #row = [annotation_file.split(os.sep)[-1]] + row
                writer.writerow(row)

        logger.info('Written annotations')

    def __resize_annotation(self, f):
        tree = etree.parse(f)
        for dim in tree.xpath("size"):

            width = self.__get_int_value(dim.xpath("width")[0].text)
            height = self.__get_int_value(dim.xpath("height")[0].text)

        for dim in tree.xpath("object/bndbox"):
            xmin = self.__get_int_value(dim.xpath("xmin")[0].text) / (width / IMAGE_SIZE)
            ymin = self.__get_int_value(dim.xpath("ymin")[0].text) / (height / IMAGE_SIZE)
            xmax = self.__get_int_value(dim.xpath("xmax")[0].text) / (width / IMAGE_SIZE)
            ymax = self.__get_int_value(dim.xpath("ymax")[0].text) / (height / IMAGE_SIZE)
        return [int(xmax), int(ymax), int(xmin), int(ymin)]

    def __get_int_value(self, text_value):
        return int(text_value.split('.')[0])