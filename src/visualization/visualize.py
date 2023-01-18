from typing import List, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

from tensorflow.python.framework.ops import Tensor


def show_image(
    image: Union[Tensor, np.ndarray],
    segmentations: Union[Polygon, List[Polygon]] = None,
    alpha: float = 1,
    fill: bool = False
):

    if isinstance(image, Tensor):
        image = image.numpy().astype(np.uint8)

    if segmentations is None:
        segmentations = []

    if isinstance(segmentations, Polygon):
        segmentations = [segmentations]

    __show_image__(
        image,
        segmentations,
        alpha,
        fill
    )


def __show_image__(
    image: np.ndarray,
    segmentations: List[Polygon] = None,
    alpha: float = 1,
    fill: bool = False
):
    image_copy = image.copy()

    for segmentation in (segmentations or []):

        def int_coords(coords):
            return np.array(coords).round().astype(np.int32)

        overlay = image_copy.copy()

        if fill:
            exterior = [int_coords(segmentation.exterior.coords)]
            cv2.fillPoly(overlay, exterior, color=(255, 0, 0))
            cv2.addWeighted(overlay, alpha, image_copy, 1 - alpha, 0, image_copy)
        else:
            pt1 = (int(segmentation.bounds[0]), int(segmentation.bounds[1]))
            pt2 = (int(segmentation.bounds[2]), int(segmentation.bounds[3]))
            cv2.rectangle(image_copy, pt1, pt2, color=(255, 0, 0))

    plt.figure()
    plt.imshow(image_copy)
