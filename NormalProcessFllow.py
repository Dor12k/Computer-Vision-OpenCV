import cv2

cap = cv2.VideoCapture(0)
# This is preprocessing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# The main loop
while True:
    _, frame = cap.read()
    # This is the processing
    frame = cv2.resize(frame, (4, 5))
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame[0, 0] = 255

    print(frame.shape)
    print(frame.ndim)
    print(frame.dtype)
    print(frame)

    break
    # Here we show the image in a window
    cv2.imshow("Webcam", frame)

    # Check if q was pressed
    if cv2.waitKey(1) == ord('q'):
        break


