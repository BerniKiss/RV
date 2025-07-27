# PRESS Q TO EXIT
import cv2
import numpy as np

# cascadeclassifier: recognises faces
# cv2.data.haarcascades just the path
# haarcascade_frontalface_default.xml the modell from the FRONT FACE PART
face_obj = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
print("Cascade falj location:", cv2.data.haarcascades)

# Load smiley
# RGB + 4TH alfa = transaprent
smiley = cv2.imread('Face/evil.png', cv2.IMREAD_UNCHANGED)
# smiley = cv2.imread('smile.png', cv2.IMREAD_COLOR)
print(smiley.shape)

def overlay_image_alpha(img, img_rgb, pos, alfa):
    x, y = pos

    # img size
    h, w = img_rgb.shape[0], img_rgb.shape[1]

    # put
    emoji = img[y:y+h, x:x+w]

    # where the mask is not 0
    img_noBg = cv2.bitwise_and(emoji, emoji, mask=cv2.bitwise_not(alfa))
    img_fg = cv2.bitwise_and(img_rgb, img_rgb, mask=alfa)

    final = cv2.add(img_noBg, img_fg)
    img[y:y+h, x:x+w] = final


# open camera
vid = cv2.VideoCapture(0)

while True:
    # every few seconds new frame, one frame is like a picture
    # new = if there is frame
    # frame = a numpy array, a matrix
    new, frame = vid.read()
    '''
    for y in range(5):
        for x in range(5):
            pixel = frame[y, x]
            print(f"Pixel koordinates: (x={x}, y={y}), COLOUJR (BGR): {pixel}")
    '''
    if not new:
        break

    # detect face
    # scaleFactor: the searching square sizeing
    # third param: little squares, in a frame
    # 30% smaller
    faces = face_obj.detectMultiScale(frame, 1.3, 5)

    # x,y = left
    # w = from x to right
    # h = from y down
    for (x, y, w, h) in faces:

        smiley_resized = cv2.resize(smiley, (w, h))
        smiley_rgb = smiley_resized[:, :, :3]
        alfa = smiley_resized[:, :, 3]

        if x + w <= frame.shape[1] and y + h <= frame.shape[0]:
            overlay_image_alpha(frame, smiley_rgb, (x, y), alfa)

    cv2.imshow("Face with Smiley", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
