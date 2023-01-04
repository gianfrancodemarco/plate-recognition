import cv2
import numpy as np


def get_raw_image(img):
    return img


def get_bw_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def get_edge_detection_image(img):
    # converting to gray scale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # remove noise
    img = cv2.GaussianBlur(img, (11, 11), 1)

    # convolute with proper kernels
    img = cv2.Laplacian(img, cv2.CV_32F, ksize=3)

    return img


def get_canny_img(img):
    # remove noise
    img = cv2.GaussianBlur(img, (5, 5), 1)

    img = cv2.Canny(image=img, threshold1=120, threshold2=200)  # Canny Edge Detection

    return img


def get_blurry_img(img):
    img = cv2.GaussianBlur(img, (5, 5), 1)

    return img


def change_image_brightness(image, delta):
    return image + delta


def augment_dataset(X, y):
    X_blurry = list(map(get_blurry_img, X))
    X_lighter = list(map(lambda img: change_image_brightness(img, 30), X))
    X_darker = list(map(lambda img: change_image_brightness(img, -30), X))
    X = np.concatenate([X, X_blurry, X_lighter, X_darker])

    # y doesn't change because these transformation doesn't change the bbox position
    y = np.concatenate([y, y, y, y])

    return X, y
