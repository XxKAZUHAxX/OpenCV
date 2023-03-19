import numpy as np
import matplotlib.pyplot as plt
import cv2



pic = cv2.imread("pics/omen.jpg")

pic_red = pic.copy()
pic_red[:, :, 1] = 0
pic_red[:, :, 2] = 0
pic_red = cv2.cvtColor(pic_red, cv2.COLOR_BGR2RGB)


cv2.imshow('Picture',pic_red)
cv2.waitKey()
