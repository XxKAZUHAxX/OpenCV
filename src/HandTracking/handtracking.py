import cv2
import mediapipe as mp
import time
import hand_tracking_module as htm

cap = cv2.VideoCapture(0)
cap.set(3, 1920) # Screen width
cap.set(4, 1080) # Screen height
cap.set(10, 130) # Screen Brightness
pTime = 0
cTime = 0
detector = htm.handDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
            print(lmList[8])

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, "FPS: " + str(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break