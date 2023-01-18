import random

import cv2
import numpy as np


def get_raw_image(image):
    return image


def random_get_blurry_image(image):
    ksize = random.choice([1, 3])
    sigma = random.randint(1, 3)
    return get_blurry_img(image, ksize, sigma)


def get_blurry_img(image, ksize: int, sigma: int):
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def random_change_image_brightness(image):
    delta = random.randint(-75, 75)
    return change_image_brightness(image, delta)


def change_image_brightness(image, delta: int):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # convert it to hsv
    h, s, v = cv2.split(hsv)

    v = cv2.add(v, delta)
    v[v > 255] = 255
    v[v < 0] = 0

    final_hsv = cv2.merge((h, s, v))
    image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)

    return image


def random_image_augmentation(image: np.ndarray):
    """
    Perform a position invariant transformation to the image
    The transformation is randomly chosen among:
    - no transformation
    - blurry the image
    - change the image brightness
    """

    functions = [get_raw_image, random_get_blurry_image, random_change_image_brightness]
    random_function = random.choice(functions)
    return random_function(image)
