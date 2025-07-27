import cv2
import numpy as np

color_ranges = {
    'Red': (np.array([0, 50, 50]), np.array([10, 255, 255])),
    'Green': (np.array([38, 50, 50]), np.array([78, 255, 255])),
    'Blue': (np.array([78, 50, 50]), np.array([131, 255, 255]))
}


color_colors = {
    'Red': (0, 0, 255),
    'Green': (0, 255, 0),
    'Blue': (255, 0, 0)
}

def process_frame(frame):
    display_frame = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for color, (lower_bound, upper_bound) in color_ranges.items():
        mask = cv2.inRange(hsv, lower_bound, upper_bound)


        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)

                cv2.rectangle(display_frame, (x, y), (x + w, y + h),
                            color_colors[color], 3)

                cv2.putText(display_frame, color, (x, y-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_colors[color], 2)

    return display_frame


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break


    processed_frame = process_frame(frame)


    cv2.putText(processed_frame, "Color Tracker: Red, Green, Blue", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


    cv2.imshow('Color Tracker', processed_frame)


    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()