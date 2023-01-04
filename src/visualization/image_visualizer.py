from typing import List, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon


def show_image(
    image,
    segmentations: Union[Polygon, List[Polygon]] = [],
    alpha: float = 0.2
):
    image_copy = image.copy()

    if isinstance(segmentations, Polygon):
        segmentations = []

    for segmentation in segmentations:

        def int_coords(coords):
            return np.array(coords).round().astype(np.int32)

        exterior = [int_coords(segmentation.exterior.coords)]
        overlay = image_copy.copy()
        cv2.fillPoly(overlay, exterior, color=(255, 0, 0))
        cv2.addWeighted(overlay, alpha, image_copy, 1 - alpha, 0, image_copy)

    plt.figure()
    plt.imshow(image_copy)
    plt.savefig('prediction.jpg')
    