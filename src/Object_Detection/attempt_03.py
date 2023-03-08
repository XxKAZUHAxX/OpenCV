import cv2
import numpy as np

<<<<<<< HEAD
thres = 0.5
nms_threshold = 0.2
cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)
cap.set(10,150)


classNames = []
classFile = 'imports/coco.names'

with open(classFile) as f:
    classNames = f.read().rstrip('\n').split()
    

configPath = 'imports/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'imports/frozen_inference_graph.pb'


net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


while True:
    success, img = cap.read()
    classIds, confs, bboxs = net.detect(img, confThreshold=thres)
    # print(classIds, bboxs)
    bboxs = list(bboxs)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))

    indices = cv2.dnn.NMSBoxes(bboxs, confs, thres, nms_threshold)
    print(indices)

    for i in indices:
        box = bboxs[i]
        x,y,w,h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x+w, y+h), color = (0,255,0), thickness = 2)
        cv2.putText(img, classNames[classIds[i]-1].upper(), (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)
    
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    

=======
# img = cv2.imread('pics/Nadia_Murad.jpg')
# img = cv2.resize(img, (880, 1246))

cap = cv2.VideoCapture(0)
cap.set(3,1920)
cap.set(4,1080)


weightsPath = 'imports/frozen_inference_graph.pb'
configPath = 'imports/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

classNames = []
with open('imports/coco.names', 'r') as f:
    classNames = f.read().strip().split()

np.random.seed(42)
color = np.random.randint(0, 255, size=(len(classNames), 3))

while True:
    success, img = cap.read()
    h = img.shape[0]
    w = img.shape[1]

    blob = cv2.dnn.blobFromImage(img, 1.0/127.5, (320, 320), [127.5,127.5,127.5])
    net.setInput(blob)
    output = net.forward()
    print(output.shape)


    for detection in output[0, 0, :, :]:
        probability = detection[2]
        if probability < 0.5:
            continue
        
        box = [int(a * b) for a,b in zip(detection[3:7], [w, h, w, h])]
        box = tuple(box)
        cv2.rectangle(img, box[:2], box[2:], (0, 255, 0), 2)
        classId = int(detection[1])
        if output.shape[0] <= len(classNames):
            label = f"{classNames[classId - 1]} {probability * 100:.2f}"
            cv2.putText(img, label.upper(), (box[0]+10, box[1]+50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
>>>>>>> da25fda (First Commit)
