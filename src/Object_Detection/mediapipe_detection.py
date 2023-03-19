import cv2
import mediapipe as mp
import time

mp_objectron = mp.solutions.objectron
mp_draw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
pTime = 0

with mp_objectron.Objectron(static_image_mode = False,
                            max_num_objects = 2,
                            min_detection_confidence = 0.5,
                            min_tracking_confidence = 0.7,
                            model_name = 'Cup') as objectron:
    while cap.isOpened():
        success, frame = cap.read()

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(frame, f"FPS: {int(fps)}", (20,70), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = objectron.process(frame)
        if results.detected_objects:
            for detected_object in results.detected_objects:
                mp_draw.draw_landmarks(frame, detected_object.landmarks_2D, mp_objectron.BOX_CONNECTIONS)
                mp_draw.draw_axis(frame, detected_object.rotation, detected_object.translation)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imshow('Mediapipe Objectron', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
