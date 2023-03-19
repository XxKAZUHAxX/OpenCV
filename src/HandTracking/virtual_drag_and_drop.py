import cv2
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector
import time

cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)
detector = HandDetector(maxHands=1, detectionCon=0.7)
color_rectangle = 255, 0 ,255
pTime = 0
cx, cy, w, h = 100, 100, 200, 200

while cap.isOpened():
    success, img = cap.read()
    img = cv2.flip(img, 1)
    lmList = detector.findHands(img)
    lmPos = [i['lmList'] for i in lmList[0]]

    # if len(lmPos)!= 0:
        # print(lmPos[0][0])
    
    if len(lmList[0])!= 0:
        length, _ , _ = detector.findDistance(lmPos[0][8][:2], lmPos[0][12][:2], img)
        print(length)
        if length < 50:

            cursor = [i['lmList'][8] for i in lmList[0]] # Accessing a specific index position
            # print(cursor[0])
            
            if cx-w//2 < cursor[0][0] < cx+w//2 and cy-h//2 < cursor[0][1] < cy+h//2:
                color_rectangle = (0, 255, 0)
                cx, cy = cursor[0][0], cursor[0][1]
            else:
                color_rectangle = (255, 0, 255)
    
    cv2.rectangle(img, (cx-w // 2, cy-h // 2), (cx+w // 2, cy+h // 2), (color_rectangle), -1)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, "FPS: " + str(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
