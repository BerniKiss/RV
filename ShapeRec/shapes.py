import cv2
import numpy as np

img = cv2.imread('ShapeRec/shapes.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def get_shape_label(sides):
    if sides == 3:
        return 'Triangle'
    elif sides == 4:
        return 'Rectangle'
    elif sides == 5:
        return 'Pentagon'
    elif sides == 6:
        return 'Hexagon'
    else:
        return 'Circle'

for i, contour in enumerate(contours):
    if i == 0:  # skip backround
        continue

    area = cv2.contourArea(contour)
    if area < 200:
        continue

    approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)

    cv2.drawContours(img, [contour], 0, (255, 0, 0), 2)

    # center
    M = cv2.moments(contour)
    if M['m00'] != 0:
        x = int(M['m10'] / M['m00'])
        y = int(M['m01'] / M['m00'])
    sides = len(approx)

    label = get_shape_label(sides)
    cv2.putText(img, label, (x-50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imshow('shapes', img)
cv2.waitKey(0)
cv2.destroyAllWindows()