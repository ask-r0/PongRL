import cv2
import numpy as np


class ImagePreProcessor:
    def __init__(self, initial_height, initial_width, crop_pixels_top, crop_pixels_bottom, do_gray, do_crop):
        self.initial_height = initial_height
        self.initial_width = initial_width

        self.crop_pixels_top = crop_pixels_top
        self.crop_pixels_bottom = crop_pixels_bottom

        self.processed_height = initial_height - crop_pixels_top - crop_pixels_bottom
        self.processed_width = initial_width

        self.do_gray = do_gray
        self.do_crop = do_crop

    def get_processed_img(self, img):
        if self.do_gray:
            img = self.__img_to_gray(img)

        img = self.__img_crop_height(img)
        return img

    def get_black_image(self):
        return np.zeros((self.processed_height, self.processed_width))

    def __img_crop_height(self, img):
        return img[self.crop_pixels_top:(img.shape[0] - self.crop_pixels_bottom), :]

    def __img_to_gray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
