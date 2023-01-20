import builtins  # Because importing all from keras overrides builtins.min
import logging
from functools import lru_cache

import tensorflow as tf
from keras.backend import cast, clip, epsilon, maximum, minimum
from tensorflow import abs, float32, transpose


def iou(y_true, y_pred):

    tf.print("y true:", y_true)
    tf.print("y pred:", y_pred)

    # AOG = Area of Groundtruth box
    AoG = abs(transpose(y_true)[0] - transpose(y_true)[2] + 1) * \
        abs(transpose(y_true)[1] - transpose(y_true)[3] + 1)

    # tf.print('aog', AoG)

    # AOP = Area of Predicted box
    AoP = abs(transpose(y_pred)[0] - transpose(y_pred)[2] + 1) * \
        abs(transpose(y_pred)[1] - transpose(y_pred)[3] + 1)

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


def lev_dist(a, b):
    '''
    This function will calculate the levenshtein distance between two input
    strings a and b

    params:
        a (String) : The first string you want to compare
        b (String) : The second string you want to compare

    returns:
        This function will return the distnace between string a and b.

    example:
        a = 'stamp'
        b = 'stomp'
        lev_dist(a,b)
        >> 1.0
    '''

    if not a:
        return len(b)

    if not b:
        return len(a)

    @lru_cache(None)  # for memorization
    def min_dist(s1, s2):

        if s1 == len(a) or s2 == len(b):
            return len(a) - s1 + len(b) - s2

        # no change required
        if a[s1] == b[s2]:
            return min_dist(s1 + 1, s2 + 1)

        return 1 + builtins.min(
            min_dist(s1, s2 + 1),  # insert character
            min_dist(s1 + 1, s2),  # delete character
            min_dist(s1 + 1, s2 + 1),  # replace character
        )

    return min_dist(0, 0)