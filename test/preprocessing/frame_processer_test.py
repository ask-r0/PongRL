from src.frame_processer import get_processed_img
import cv2

img = cv2.imread("game_img.png")
cv2.imwrite("result.png", get_processed_img(img))
