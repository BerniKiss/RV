import cv2

# original picture load
image = cv2.imread("showPoints/nature.jpg")
if image is None:
    print("Error: Img can not be found!")
    exit()

# read points from file
points = []
with open("showPoints/coord.txt", "r") as file:
    for line in file:
        x_str, y_str = line.strip().split()
        x, y = int(float(x_str)), int(float(y_str))
        points.append((x, y))

# draw points
for x, y in points:
    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # green point

# drae line between the points
for i in range(1, len(points)):
    cv2.line(image, points[i - 1], points[i], (255, 0, 0), 2)  # blue line

# 5. show pic
cv2.imshow("Points shown", image)
cv2.waitKey(0)
cv2.destroyAllWindows()