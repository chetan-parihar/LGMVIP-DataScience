#convert Image into Sketch

import cv2

image=cv2.imread("sir.jpg")

gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

invert = 255 - gray_image

blurr = cv2.GaussianBlur(invert, (21, 21), 0)

invert = 255 - blurr

pencil_sketch = cv2.divide(gray_image, invert, scale=256.0)

cv2.imshow("original image", image)
cv2.imshow("invert",invert)
cv2.imshow("pencil sketch", pencil_sketch)
cv2.waitKey(0)

