import cv2
import mediapipe as mp
import numpy as np
import time
import hand_tracking_module as htm
import math
import cvzone.HandTrackingModule

cap = cv2.VideoCapture(0)
cap.set(3, 1980) # Screen width
cap.set(4, 1080) # Screen height
cap.set(10, 130) # Screen Brightness
pTime = 0

detector = htm.handDetector()

while True:
    success, img = cap.read()

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList)!= 0:
        # print(lmList[4], lmList[8])

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx ,cy = (x1+x2)//2, (y1+y2)//2
        
        cv2.circle(img, (x1, y1), 15, (0, 255, 0), -1)
        cv2.circle(img, (x2, y2), 15, (0, 255, 0), -1)
        cv2.circle(img, (cx, cy), 15, (0, 255, 0), -1)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
        length = math.hypot(x1-x2, y1-y2)
        if length < 50:
            cv2.circle(img, (cx, cy), 15, (255, 0, 0), -1)
        # print(length)
    
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()