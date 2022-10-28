import cv2


def img_to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def img_crop_height(img, pixels_from_top, pixels_from_bottom):
    return img[pixels_from_top:img.shape[0] - pixels_from_bottom, :]


def get_processed_img(img, to_gray=True, pixels_from_top=25, pixels_from_bottom=5):
    if to_gray:
        img = img_to_gray(img)

    img = img_crop_height(img, pixels_from_top, pixels_from_bottom)
    return img


