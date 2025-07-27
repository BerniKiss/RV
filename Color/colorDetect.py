
import cv2
import numpy as np

selected_color = None
lower_bound = None
upper_bound = None
last_selected_color = None


def click_event(event, x, y, flags, param):
    global selected_color, lower_bound, upper_bound, last_selected_color,color_name

    if event == cv2.EVENT_LBUTTONDOWN:
        selected_color = param[y, x]
        print(f"Selected Color (BGR): {selected_color}")

        if last_selected_color is None or not np.array_equal(selected_color, last_selected_color):
            last_selected_color = selected_color
            print(f"Updated Color (BGR): {selected_color}")

            # convert to hsv
            hsv_color = cv2.cvtColor(selected_color, cv2.COLOR_BGR2HSV)
            print(f"Selected Color (HSV): {hsv_color}")

            # RANGE OF THE COLOR IN HSV
            lower_bound, upper_bound = get_color_range(hsv_color)
            print(f"Color range: Lower={lower_bound}, Upper={upper_bound}")

            # DETECT THE COLOR
            color_name = detect_color_name(hsv_color)
            print(f"Detected Color: {color_name}")
            print("-" * 40)


def get_color_range(hsv_color):
    h, s, v = hsv_color

    if s < 30 and v > 200:
        # white
        return np.array([0, 0, 200]), np.array([179, 30, 255])

    # black
    if v < 50:
        return np.array([0, 0, 0]), np.array([179, 255, 50])

    if s < 30:
        # graz
        return np.array([0, 0, max(0, v-50)]), np.array([179, 30, min(255, v+50)])

    if 0 <= h <= 10 or 170 <= h <= 179:
        # RED
        return np.array([0, 50, 50]), np.array([15, 255, 255])
    elif 10 < h <= 22:
        # ORANGE
        return np.array([10, 50, 50]), np.array([22, 255, 255])
    elif 22 < h <= 38:
        # YELLOW
        return np.array([22, 50, 50]), np.array([38, 255, 255])
    elif 38 < h <= 78:
        # GREEN
        return np.array([38, 50, 50]), np.array([78, 255, 255])
    elif 78 < h <= 131:
        # BLUE
        return np.array([78, 50, 50]), np.array([131, 255, 255])
    elif 131 < h <= 170:
        # PURPLE
        return np.array([131, 50, 50]), np.array([170, 255, 255])

    # OTHER BELNDIG COLORS
    r = 15
    lower_h = max(0, h - r)
    upper_h = min(179, h + r)

    return np.array([lower_h, 50, 50]), np.array([upper_h, 255, 255])


def detect_color_name(hsv_color):
    h, s, v = hsv_color
    print(f"HSV values: H={h}, S={s}, V={v}")

    if s < 30 and v > 200:
        return "White"

    if v < 50:
        return "Black"

    if s < 30:
        return "Gray"

    if s >= 50:
        if 0 <= h <= 10 or 170 <= h <= 179:
            return "Red"
        elif 10 < h <= 22:
            return "Orange"
        elif 22 < h <= 38:
            return "Yellow"
        elif 38 < h <= 78:
            return "Green"
        elif 78 < h <= 131:
            return "Blue"
        elif 131 < h <= 170:
            return "Purple"

    return f"Unknown (H:{h}, S:{s}, V:{v})"

# creating test image
def create_test_image():
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    colors = [
        (0, 0, 255),      # red
        (0, 255, 0),      # green
        (255, 0, 0),      # blue
        (0, 255, 255),    # zellow
        (255, 0, 255),    # magenta
        (255, 255, 0),    # cyan
        (0, 165, 255),    # orange
        (128, 0, 128),    # purple
        (255, 255, 255),  # white
    ]

    for i, color in enumerate(colors):
        row = i // 3
        col = i % 3
        x1, y1 = col * 200, row * 133
        x2, y2 = x1 + 200, y1 + 133
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    return img


frame = cv2.imread('Color/colour.jpg')
if frame is None:
    print("img can not be found, creating image..")
    frame = create_test_image()

cv2.namedWindow('Color Tracker')
cv2.setMouseCallback('Color Tracker', click_event, param=frame)

print("Click on a color. 'q' fro exit")

while True:
    display_frame = frame.copy()

    if selected_color is not None and lower_bound is not None and upper_bound is not None:
        #  1. CONVERT TO HSV FROM BGR
        hsv = cv2.cvtColor(display_frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        kernel = np.ones((3,3), np.uint8)

        mask = cv2.dilate(mask, kernel, iterations=1)
        res = cv2.bitwise_and(display_frame, display_frame, mask=mask)
    else:
        res = display_frame
    cv2.putText(res, "Click for naming color", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)

    if selected_color is not None:
        hsv_color = cv2.cvtColor(np.uint8([[selected_color]]), cv2.COLOR_BGR2HSV)[0][0]
        color_name = detect_color_name(hsv_color)
        color_text = f"Detected: {color_name}"
        cv2.putText(res, color_text, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)


    height, width = res.shape[:2]
    max_width = 1000
    if width > max_width:
        aspect_ratio = max_width / width
        new_width = max_width
        new_height = int(height * aspect_ratio)
        display_frame = cv2.resize(display_frame, (new_width, new_height))
    cv2.imshow('Color Tracker', display_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()