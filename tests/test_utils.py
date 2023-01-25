from random import random

import numpy as np
import tensorflow as tf
from src.utils import set_random_states


class TestUtils:
    
    def test_set_random_states(self):
        
        set_random_states(69)
        assert random() == 0.6842409524120733
        assert list(tf.random.normal([3])) == [-1.1153845 ,  0.12418792,  0.73649114]
        assert np.random.random() == 0.29624916167243354

        set_random_states(90)
        assert random() == 0.20367044742105156
        assert list(tf.random.normal([3]))== [1.6884607 ,  -0.25445077,  0.07089669]
        assert np.random.random() == 0.1530541987661469
