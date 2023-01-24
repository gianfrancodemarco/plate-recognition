import albumentations as A
import cv2
import numpy as np


def augment_image(image: np.ndarray, bbox: np.ndarray): 
    transform = A.Compose([
        A.RandomCrop(p=0.2, width=200, height=200),
        A.HorizontalFlip(p=0.2),
        A.Blur(p=0.1),
        A.CLAHE(p=0.1),
        A.Equalize(p=0.3),
        A.ColorJitter(p=0.1),
        A.RandomShadow(p=0.3),
        A.RandomBrightness(p=0.1),
        A.ShiftScaleRotate(p=0.2, rotate_limit=15)
    ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=1, label_fields=[]))

    transformed = transform(image=image, bboxes=[bbox])
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']

    if len(transformed_bboxes) == 0:
        return image, bbox

    return transformed_image, transformed_bboxes[0]

def preprocess_image(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    return image


def crop_image(image, bbox):
    _cropped_image = image.copy()
    _cropped_image = _cropped_image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
    return _cropped_image
