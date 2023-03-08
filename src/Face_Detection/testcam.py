import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 1920)

while True:
    success, img = cap.read()
    cv2.imshow("Webcam", img)
    if cv2.waitKey(10) & 0xFF == 27:
        break
