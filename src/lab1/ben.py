import numpy as np
import cv2 as cv

img = cv.imread('/home/martin/turtlebot3_ws/src/lab1/ben.jpg')

assert img is not None, "file could not be read, check with os.path.exists()"

cv.imshow('ben', img)
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('grey_ben',gray_img)
cv.waitKey(0)
cv.destroyAllWindows()