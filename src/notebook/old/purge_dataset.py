import random
from matplotlib import pyplot as plt
import cv2
import numpy as np
from src.datasets.AnnotatedImageVisualizer import AnnotatedImageVisualizer


def purge_dataset():
    X = []
    y = []

    annotations = pd.read_csv(annotations_path)
    new_annotations = annotations.copy(deep=True)

    resume_from = 350
    for idx, image in enumerate(annotations['name'].tolist()[resume_from:]):
        img = cv2.imread(os.path.join(images_path, str(image) + '.jpg'))
        ann = annotations.drop(annotations.columns[[0]], axis=1).iloc[idx + resume_from]

        AnnotatedImageVisualizer().show_image(img, ann)
        delete = input(f"Do you want to delete this image? N. {idx + resume_from} \n")

        if delete == 'y':
            os.remove(os.path.join(gdrive_path, 'images', str(image) + '.jpg'))
            os.remove(os.path.join(images_path, str(image) + '.jpg'))
            new_annotations.drop(new_annotations[new_annotations.name == image].index, inplace=True)
            print(new_annotations.shape[0])

            with open(annotations_path, 'w') as f:
                new_annotations.to_csv(f, index=False)

            with open(os.path.join(gdrive_path, 'annotations.csv'), 'w') as f:
                new_annotations.to_csv(f, index=False)

purge_dataset()