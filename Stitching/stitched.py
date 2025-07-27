import numpy as np
import cv2
import glob
import imutils

print(cv2.__version__)
print(hasattr(cv2, 'Stitcher_create'))
img_paths = glob.glob('Stitching/unstitchedImg/*.jpg')
images = []

for image in img_paths:
    img = cv2.imread(image)
    images.append(img)
    img_resized = cv2.resize(img, (800, 800))
    cv2.imshow("Img",img_resized)
    cv2.waitKey(0)

print(f"BRead images: {len(images)}")
for i, img in enumerate(images):
    print(f"Img {i+1} size: {img.shape}")


imageStitcher = cv2.Stitcher_create()

error, stitched_img =imageStitcher.stitch(images)


if not error:
    print('ide nem jut el')
    cv2.imwrite("Stitching/stitchedOutP.jpg", stitched_img)
    print('Image created successfully!')

    stitched_img = cv2.copyMakeBorder(stitched_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0,0,0))

    # to find the contour we needgray img
    # gray = for every pixel there willl be only 1 number
    gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)

    # makes the image white and blacvk
    thresh_img = cv2.threshold(gray, 0, 255 , cv2.THRESH_BINARY)[1]

    thresh_img_r = cv2.resize(thresh_img, (800, 800))
    cv2.imshow("Threshold Image", thresh_img_r)
    cv2.waitKey(0)

    # RETR_EXTERNAL = only the contour, does not matter if in the picture
    contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    contours = imutils.grab_contours(contours)

    # biggest area of interest
    areaOI = max(contours, key=cv2.contourArea)

    cv2.drawContours(stitched_img, [areaOI], -1, (0, 255, 0), 3)

    # a black empty img
    mask = np.zeros(thresh_img.shape, dtype="uint8")

    # calculates where the rectange has to be on the black empy image
    x, y, w, h = cv2.boundingRect(areaOI)

    # on the empty mask it will draw a rectagle
    cv2.rectangle(mask, (x,y), (x + w, y + h), 255, -1)

    # i will use this later
    minRectangle = mask.copy()
    sub = mask.copy()

    # this part shortens until there will be no black part isnide the rectangle
    while cv2.countNonZero(sub) > 0:
        minRectangle = cv2.erode(minRectangle, None)
        sub = cv2.subtract(minRectangle, thresh_img)


    contours = cv2.findContours(minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv2.contourArea)

    minRectangle_r = cv2.resize(minRectangle, (800, 800))
    cv2.imshow("minRectangle Image", minRectangle_r)
    cv2.waitKey(0)

    # what parst need to be cut out
    x, y, w, h = cv2.boundingRect(areaOI)

    # the img cut out
    stitched_img = stitched_img[y:y + h, x:x + w]

    cv2.imwrite("Stitching/stitchedOutProcessed.png", stitched_img)

    stitched_img_r = cv2.resize(stitched_img, (800, 800))
    cv2.imshow("Stitched Processed", stitched_img_r)

    cv2.waitKey(0)