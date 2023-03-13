import cv2
import mediapipe as mp
import numpy as np
import time
import hand_tracking_module as htm

cap = cv2.VideoCapture(0)
cap.set(3, 1980)
cap.set(4, 1080)
pTime = 0

detector = htm.handDetector()

while True:
    success, img = cap.read()

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList)!= 0:
        print(lmList[4])
    
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()