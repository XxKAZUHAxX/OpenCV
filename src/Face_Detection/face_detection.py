import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection.FaceDetection()
mp_drawing = mp.solutions.drawing_utils



cam = cv2.VideoCapture(0)

cam.set(3, 1920)
cam.set(4, 1080)

while cam.isOpened():

    success, img = cam.read()

    # detect faces using mediapipe

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_face_detection.process(img)
<<<<<<< HEAD
    
=======
    print(results)
>>>>>>> da25fda (First Commit)
    # draw face detect annotations

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(img,detection)

    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
