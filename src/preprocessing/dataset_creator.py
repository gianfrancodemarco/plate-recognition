import os
import logging
import pandas as pd
import shutil
import csv

# create logger
logger = logging.getLogger(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))


class DatasetCreator:
    DATASETS_PATH = os.path.join(dir_path, '..', '..', 'resources')
    OUTPUT_PATH = os.path.join(dir_path, '..', '..', 'final_dataset')
    IMAGES_OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'images')
    NOT_ANNOTATED_IMAGES_OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'not_annotated_images')
    ANNOTATIONS_OUTPUT_PATH = OUTPUT_PATH

    def create_dataset(self):

        logger.info("Running create dataset")

        if not os.path.exists(self.OUTPUT_PATH):
            os.makedirs(self.OUTPUT_PATH)
            logger.info("Created final_dataset folder")

        if not os.path.exists(self.IMAGES_OUTPUT_PATH):
            os.makedirs(self.IMAGES_OUTPUT_PATH)
            logger.info("Created images folder")

        if not os.path.exists(self.NOT_ANNOTATED_IMAGES_OUTPUT_PATH):
            os.makedirs(self.NOT_ANNOTATED_IMAGES_OUTPUT_PATH)
            logger.info("Created not annotated images folder")

        annotations_file = os.path.join(self.ANNOTATIONS_OUTPUT_PATH, 'annotations.csv')

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

            idx = 0
            for dataset in os.listdir(self.DATASETS_PATH):

                images_path = os.path.join(self.DATASETS_PATH, dataset, 'images')
                images = os.listdir(images_path)

                source_annotations_path = os.path.join(self.DATASETS_PATH, dataset, 'annotations.csv')

                try:
                    source_annotations = pd.read_csv(source_annotations_path)
                except:
                    source_annotations = pd.DataFrame()

                for image in images:
                    if source_annotations.empty:
                        shutil.copyfile(os.path.join(images_path, image), os.path.join(self.NOT_ANNOTATED_IMAGES_OUTPUT_PATH, f'{idx}.jpg'))
                    else:
                        shutil.copyfile(os.path.join(images_path, image), os.path.join(self.IMAGES_OUTPUT_PATH, f'{idx}.jpg'))
                        name = int(image.split('.')[0])
                        source_row = source_annotations[source_annotations['name'] == name]
                        row = [idx, source_row['xmin'][name], source_row['ymin'][name], source_row['xmax'][name], source_row['ymax'][name]]
                        writer.writerow(row)

                    idx += 1

        logger.info("Done create dataset")
