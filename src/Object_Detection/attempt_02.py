import cv2
import numpy as np

thres = 0.7
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
    