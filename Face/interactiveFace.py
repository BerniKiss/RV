import cv2
import numpy as np
import os

EMOJI_PATH = "Face/emojis"
EMOJI_SIZE = 30

# load emojis
emoji_files = [f for f in os.listdir(EMOJI_PATH)]
emojis = [cv2.imread(os.path.join(EMOJI_PATH, f), cv2.IMREAD_UNCHANGED) for f in emoji_files]
selected_emoji_idx = 0
current_effect = "none"

def apply_effect(frame):
    if current_effect == "blur":
        return cv2.GaussianBlur(frame, (15, 15), 0)
    elif current_effect == "gray":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif current_effect == "edges":
        return cv2.Canny(frame, 100, 200)
    return frame

def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    x, y = pos
    h, w = img_overlay.shape[:2]
    if y + h > img.shape[0] or x + w > img.shape[1]:
        return

    emoji = img[y:y+h, x:x+w]
    img_bg = cv2.bitwise_and(emoji, emoji, mask=cv2.bitwise_not(alpha_mask))
    img_fg = cv2.bitwise_and(img_overlay, img_overlay, mask=alpha_mask)
    final = cv2.add(img_bg, img_fg)
    img[y:y+h, x:x+w] = final


def draw_emoji_bar(frame):
    bar_height = EMOJI_SIZE + 10
    y_base = frame.shape[0] - bar_height

    cv2.rectangle(frame, (0, y_base), (frame.shape[1], frame.shape[0]), (50, 50, 50), -1)

    # emojis next to each other
    for i, emoji in enumerate(emojis):
        x = 10 + i * (EMOJI_SIZE + 10)
        y = y_base + 5

        thumb = cv2.resize(emoji, (EMOJI_SIZE, EMOJI_SIZE))
        rgb = thumb[:, :, :3]
        alpha = thumb[:, :, 3]

        # put on the dispaly
        overlay_image_alpha(frame, rgb, (x, y), alpha)

        # if selected green
        if i == selected_emoji_idx:
            cv2.rectangle(frame, (x-2, y-2), (x+EMOJI_SIZE+2, y+EMOJI_SIZE+2), (0, 255, 0), 2)


def detect_click_on_emoji(x, y, frame_height):
    bar_top = frame_height - EMOJI_SIZE - 5
    bar_bottom = bar_top + EMOJI_SIZE

    if not (bar_top <= y <= bar_bottom):
        return -1  # if clicked not in between

    for i in range(len(emojis)):
        # calc the elft side of the emoji
        left_coord = 10 + i * (EMOJI_SIZE + 10)
        if left_coord <= x <= left_coord + EMOJI_SIZE:
            return i

    return -1


def mouse(event, x, y, flags, param):
    global selected_emoji_idx
    if event == cv2.EVENT_LBUTTONDOWN:
        idx = detect_click_on_emoji(x, y, param.shape[0])
        if idx != -1:
            selected_emoji_idx = idx


face_obj = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
vid = cv2.VideoCapture(0)
cv2.namedWindow("Emoji Cam")

while True:
    ret, frame = vid.read()
    if not ret:
        break

    frame_effect = apply_effect(frame.copy())
    if len(frame_effect.shape) == 2:
        frame_effect = cv2.cvtColor(frame_effect, cv2.COLOR_GRAY2BGR)

    faces = face_obj.detectMultiScale(frame, 1.3, 5)
    emoji = cv2.resize(emojis[selected_emoji_idx], (100, 100))
    emoji_rgb = emoji[:, :, :3]
    alpha = emoji[:, :, 3]

    for (x, y, w, h) in faces:
        emoji_resized = cv2.resize(emoji, (w, h))
        emoji_rgb = emoji_resized[:, :, :3]
        alpha = emoji_resized[:, :, 3]
        overlay_image_alpha(frame_effect, emoji_rgb, (x, y), alpha)

    draw_emoji_bar(frame_effect)
    cv2.setMouseCallback("Emoji Cam", mouse, param=frame_effect)
    cv2.imshow("Emoji Cam", frame_effect)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('b'):
        current_effect = "blur"
    elif key == ord('g'):
        current_effect = "gray"
    elif key == ord('e'):
        current_effect = "edges"
    elif key == ord('n'):
        current_effect = "none"

vid.release()
cv2.destroyAllWindows()
