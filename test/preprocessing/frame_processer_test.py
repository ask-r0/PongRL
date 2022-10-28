from src.frame_processer import ImagePreProcessor
import cv2

img = cv2.imread("game_img.png")
print(type(img))
p = ImagePreProcessor(img.shape[0], img.shape[1], 25, 5, True, True)

processed = p.get_processed_img(img)
black = p.get_black_image()

#  cv2.imwrite("processed.png", p.get_processed_img(img))
#  cv2.imwrite("black.png", p.get_black_image())
