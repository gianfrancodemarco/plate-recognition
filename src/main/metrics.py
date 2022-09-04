from tensorflow import *
from tensorflow.math import *
from tensorflow.keras.backend import *


def iou(y_true, y_pred):
    y_true = cast(y_true, float32)

    # iou as metric for bounding box regression
    # input must be as [x1, y1, x2, y2] -> i have [x2, y2, x1, y1]. 

    # AOG = Area of Groundtruth box
    AoG = abs(transpose(y_true)[0] - transpose(y_true)[2] + 1) * abs(transpose(y_true)[1] - transpose(y_true)[3] + 1)

    # tf.print('aog', AoG)

    # AOP = Area of Predicted box
    AoP = abs(transpose(y_pred)[0] - transpose(y_pred)[2] + 1) * abs(transpose(y_pred)[1] - transpose(y_pred)[3] + 1)

    # tf.print('aop', AoP)

    # overlaps are the co-ordinates of intersection box
    overlap_0 = maximum(transpose(y_true)[2], transpose(y_pred)[2])
    overlap_1 = maximum(transpose(y_true)[3], transpose(y_pred)[3])
    overlap_2 = minimum(transpose(y_true)[0], transpose(y_pred)[0])
    overlap_3 = minimum(transpose(y_true)[1], transpose(y_pred)[1])

    # intersection area
    intersection = abs(overlap_2 - overlap_0 + 1) * abs(overlap_3 - overlap_1 + 1)

    # tf.print('intersection ', intersection)

    # area of union of both boxes
    union = AoG + AoP - intersection

    # tf.print('union', union)

    # iou calculation
    iou = intersection / union

    # bounding values of iou to (0,1)
    iou = clip(iou, 0.0 + epsilon(), iou)

    return iou 