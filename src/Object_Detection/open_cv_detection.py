import cv2
import time


cap = cv2.VideoCapture('./footage/road_traffic01_1080p.mp4')
cap.set(3, 1280)
cap.set(4, 720)
# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history = 70, varThreshold=60)
pTime = 0

while True:
    success, frame = cap.read()
    h, w, _ = frame.shape
    # print(h,w)
    # Extract region of interest


    # Object Detection
    mask = object_detector.apply(frame)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 100:
            # cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y), (x + w,y + h), (0, 255, 0), 3)




    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(frame, f"FPS: {int(fps)}", (20,70), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 150, 0), 2)

    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()