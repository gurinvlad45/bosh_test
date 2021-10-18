from PIL import Image, ImageDraw
import numpy as np
import cv2


class Circle:
    def __init__(self, diameter, size=64):

        self.diameter = diameter
        if diameter >= size:
            self.diameter = size
        self.radius = self.diameter / 2
        self.size = size
        self.center_x = 32
        self.center_y = 32

    def create_circle_image(self):
        """
        Creates image based on circle diameter
        :return:
        """

        image = Image.new('L', (64, 64), 255)
        draw = ImageDraw.Draw(image)
        left_up_point = (self.center_x - self.radius, self.center_y - self.radius)
        right_down_point = (self.center_x + self.radius, self.center_y + self.radius)
        draw.ellipse([left_up_point, right_down_point], fill=True)
        img = np.array(image).astype('float32') / 255.
        # cv2.imwrite("filename.png", img * 255.0)
        return img


Circle(diameter=60).create_circle_image()
