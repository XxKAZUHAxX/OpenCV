import cv2

# img = cv2.imread('pics/Nadia_Murad.jpg')
cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)


classNames = []
with open('imports/coco.names') as f:
    classNames = f.read().rstrip('\n').split('\n')
    

configPath = 'imports/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'imports/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, confs, bboxs = net.detect(img, confThreshold=0.7)
    print(classIds, bboxs)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bboxs):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            if classId <= len(classNames):
                cv2.putText(img, classNames[classId-1].upper(), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    

