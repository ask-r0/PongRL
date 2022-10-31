import cv2
import numpy as np


class ImagePreProcessor:
    def __init__(self, out_height_width, crop_pixels_top, crop_pixels_bottom):
        self.out_height_width = out_height_width

        self.crop_pixels_bottom = crop_pixels_bottom
        self.crop_pixels_top = crop_pixels_top

    def get_processed_img(self, img):
        img = self.__img_to_gray(img)
        img = self.__img_crop_height(img)
        img = self.__img_resize(img)
        return img

    def get_black_image(self):
        return np.zeros((self.out_height_width, self.out_height_width))

    def __img_crop_height(self, img):
        return img[self.crop_pixels_top:(img.shape[0] - self.crop_pixels_bottom), :]


    def __img_resize(self, img):
        return cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)

    def __img_to_gray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
