import logging
import os
import random
from datetime import datetime

import cv2
from src import utils
from src.data.annotations.coco_annotations_manager import \
    CocoAnnotationsManager
from src.data.augmentation.template_manager import ImagesTemplatesManager
from src.features.build_features import flatten_list

CARD_IMAGES_PATH = os.path.join(utils.DATA_PATH, 'raw', 'templates')

ANNOTATIONS_PATH = os.path.join(utils.DATA_PATH, 'annotations', 'annotations.json')
TEMPLATES_CONFIG_PATH = os.path.join(utils.DATA_PATH, 'annotations', 'templates.json')
PROCESSD_PATH = os.path.join(utils.DATA_PATH, 'processed')

_, _, files = next(os.walk(os.path.join(utils.DATA_PATH, 'raw', 'card_images')))
num_card_images = len(files)


def get_random_patches(num):
    return [
        cv2.imread(os.path.join(utils.DATA_PATH, 'raw', 'card_images',
                   f'{random.randint(1, num_card_images)}_normal.jpg'))
        for
        patch
        in range(num)
    ]


def make_dataset():
    num_images_to_generate = 10_000

    annotations_manager = CocoAnnotationsManager()
    annotations_manager.load_annotations(ANNOTATIONS_PATH)
    image_templates_manager = ImagesTemplatesManager(
        templates_config_path=TEMPLATES_CONFIG_PATH,
        annotation_manager=annotations_manager,
    )
    templates = image_templates_manager.get_templates()

    
    for (dataset_split, percentage) in [("train", .7), ("test", .2), ("validation", .1)]:
        
        current = 0

        num_split_images = percentage * num_images_to_generate
        logging.warning(f"Generating {num_split_images} images for {dataset_split} set.")

        split_path = os.path.join(PROCESSD_PATH, dataset_split)
        if not os.path.exists(split_path):
            os.makedirs(split_path)

        generated_annotations_path = os.path.join(
            utils.DATA_PATH, 'annotations', f'{dataset_split}_set_annotations.json')
        generated_annotations_manager = CocoAnnotationsManager()
        generated_annotations_manager.create_empty_annotations(
            generated_annotations_path,
            categories={
                "id": 0,
                "name": "Card"
            },
            info={
                "year": 2023,
                "version": "1.0",
                "description": f"Annotations generated for syntetic data ({dataset_split} set)",
                "contributor": "Gianfranco Demarco",
                "url": "",
                "date_created": datetime.now().strftime("%Y-%d-%m %H:%M:%S")
            }
        )


        while current < num_split_images:
            
            logging.warning(f"{dataset_split} set: {current}/{num_split_images}")

            try:
                template = random.choice(templates)
                num_patches_to_generate = 0
                try:
                    num_patches_to_generate = template.max_patches
                except AttributeError:
                    num_patches_to_generate = len(template.segmentation_polygons)

                patches = get_random_patches(num=num_patches_to_generate)
                image, polygons = template.generate_image(patches)
                filename = f"{dataset_split}_split_{current}.jpeg"
                cv2.imwrite(os.path.join(split_path, filename), image)
                image_id = generated_annotations_manager.add_image(
                    filename=filename, height=image.shape[0], width=image.shape[1])
                generated_annotations_manager.add_annotation(
                    image_id=image_id,
                    category_id=0,
                    segmentation=[flatten_list([list(el) for el in polygon.exterior.coords]) for polygon in polygons],
                    bbox=[list(polygon.bounds) for polygon in polygons],
                    area=[polygon.area for polygon in polygons]
                )

                current += 1
            except:
                pass

        generated_annotations_manager.flush_annotations()


if __name__ == "__main__":
    make_dataset()
