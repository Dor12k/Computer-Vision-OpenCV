import cv2
import numpy as np

cap = cv2.VideoCapture(0)
# This is preprocessing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

logo = cv2.imread("logo.png")
size = 100
logo = cv2.resize(logo, (size, size))
gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

# The main loop
while True:
    _, frame = cap.read()
    # This is the processing
    frame = cv2.resize(frame, (640, 480))
    frame = cv2.flip(frame, 1)

    roi = frame[-size - 10:-10, -size - 10:-10]
    roi[np.where(mask)] = 0
    roi += logo

    # Here we show the image in a window
    cv2.imshow("Webcam", frame)

    # Check if q was pressed
    if cv2.waitKey(1) == ord('q'):
        break
        