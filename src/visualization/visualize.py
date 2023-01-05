from typing import List, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

from tensorflow.python.framework.ops import Tensor


def show_image(
    image: Union[Tensor, np.ndarray],
    segmentations: Union[Polygon, List[Polygon]] = [],
    alpha: float = 1,
    fill: bool = False
):

    if isinstance(image, Tensor):
        image = image.numpy().astype(np.uint8)
    
    if isinstance(segmentations, Polygon):
        segmentations = [segmentations]

    __show_image__(
        image,
        segmentations,
        alpha
    )


def __show_image__(
    image: np.ndarray,
    segmentations: List[Polygon] = [],
    alpha: float = 1,
    fill: bool = False
):
    image_copy = image.copy()

    for segmentation in segmentations:

        def int_coords(coords):
            return np.array(coords).round().astype(np.int32)

        overlay = image_copy.copy()
        
        if fill:
            exterior = [int_coords(segmentation.exterior.coords)]
            cv2.fillPoly(overlay, exterior, color=(255, 0, 0))
            cv2.addWeighted(overlay, alpha, image_copy, 1 - alpha, 0, image_copy)
        else:
            cv2.rectangle(image_copy, segmentation.bounds, color=(255, 0, 0))        

    plt.figure()
    plt.imshow(image_copy)
    plt.savefig('prediction.jpg')
    