import cv2
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_utils = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


cam = cv2.VideoCapture(0)

while cam.isOpened():
    success, img = cam.read() 

    # apply face mesh
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.FaceMesh(refine_landmarks=True).process(img)

    # apply annotations to the image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing_utils.draw_landmarks(img, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, None, mp_drawing_styles.get_default_face_mesh_tesselation_style())
            mp_drawing_utils.draw_landmarks(img, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, None, mp_drawing_styles.get_default_face_mesh_contours_style())
            mp_drawing_utils.draw_landmarks(img, face_landmarks, mp_face_mesh.FACEMESH_IRISES, None, mp_drawing_styles.get_default_face_mesh_iris_connections_style())
    
    cv2.imshow('MediaPipe Face Mesh', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
