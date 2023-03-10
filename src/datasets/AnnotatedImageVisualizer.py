from matplotlib import pyplot as plt
import cv2
import numpy as np

class AnnotatedImageVisualizer:

    GREEN = (0, 255, 0)

    def show_image(self, image, rectangle=None):

        plt.figure()
        plt.axis('off')

        if type(image) == str:
            image = np.array(cv2.imread(image))

        if rectangle is not None:
            pt1, pt2, pt3, pt4 = rectangle
            annotated_image = cv2.rectangle(image, (pt1, pt2), (pt3, pt4), self.GREEN)
        else:
            annotated_image = image

        plt.imshow(annotated_image)
        plt.show(block=False)