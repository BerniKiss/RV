import cv2

points = []
img_original = None
img_display = None

def click_event(event, x, y, flags, param):
    global img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Saved: {x} {y}")

        img_display = img_original.copy()

        for point_x, point_y in points:
            cv2.circle(img_display, (point_x, point_y), 5, (0, 0, 255), -1)

        cv2.imshow("IMG", img_display)

img_original = cv2.imread("showPoints/nature.jpg")
if img_original is None:
    print("ERROR: image can not be found")
    exit()

img_display = img_original.copy()
cv2.imshow("IMG", img_display)
cv2.setMouseCallback("IMG", click_event)

print("Click on the picture. To exit click on a random key.")
cv2.waitKey(0)
cv2.destroyAllWindows()


with open("showPoints/coord.txt", "w") as file:
    for x, y in points:
        file.write(f"{x} {y}\n")

print(f"{len(points)} points saved.")