import cv2
import numpy as np

cap = cv2.VideoCapture(0)
whT = 320
configThreshold = 0.5
nmsThreshold = 0.3

classFile = 'imports/coco.names'
classNames = []
with open(classFile, 'rt') as f:
    f.read().strip().split()


configPath = 'imports/yolov3.cfg'
weightsPath = 'imports/yolov3.weights'

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)



def FindObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > configThreshold:
                w, h = int(detection[2] * wT), int(detection[3] * hT)
                x, y = int((detection[0] * wT) - w/2), int((detection[1] * hT) - h/2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
                
    indices = cv2.dnn.NMSBoxes(bbox, confs, configThreshold, nmsThreshold)
    print(indices)
    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%' (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)



while True:
    success, img = cap.read()
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT,whT), [0,0,0], 1, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    
    # print(outputNames)
    # print(net.getUnconnectedOutLayersNames())
    FindObjects(outputs, img)
    
    cv2.imshow('Frame', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    