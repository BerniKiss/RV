import cv2
import numpy as np
import sys

# TRY FORM 1 TO 7
step = 1

if len(sys.argv) > 1:
    try:
        step = int(sys.argv[1])
    except ValueError:
        print("Need a number!")
        sys.exit(1)

img = cv2.imread("ImgProcessing/fruit.png", cv2.IMREAD_COLOR)
if img is None:
    print("ERROR: Image can not be found!")
    exit()

if step == 1:
    # normal picture
    cv2.imshow("Normal pic", img)
    cv2.waitKey(0)

# just print the pics dimension
elif step == 2:
    height, width, channels = img.shape
    print(f"Height: {height}, Width: {width}, Channels: {channels}")

   # crop a part
elif step == 3:
    points = []

    display_img = img.copy()

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Image", display_img)

            if len(points) == 4:
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]

                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                cropped = img[y_min:y_max, x_min:x_max]

                height, width = cropped.shape[:2]
                resized_cropped = cv2.resize(cropped, (width * 7, height * 7))

                cv2.imshow("Cropped & Resized", resized_cropped)
    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# grayscalin
elif step == 4:
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("graysclaing img", gray_img)
    cv2.waitKey(0)

# blur gaussian
elif step == 5:
    blur_img = cv2.GaussianBlur(img, (7,7), 0)
    cv2.imshow("img", blur_img)
    cv2.waitKey(0)

# edge detecting
elif step == 6:
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img, 100, 200)
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)

# just  add text to image
elif step == 7:
    cv2.putText(img, "Hello OpenCV", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Img with text", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
