import numpy as np
from src.visualization.visualize import show_image


class TestVisualize():
    def test_show_image(self):

        image = np.array([
            [1,1,1],
            [1,0,1],
            [1,1,1]
        ])
        show_image(image)