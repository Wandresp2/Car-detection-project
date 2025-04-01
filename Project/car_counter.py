import numpy as np
from ultralytics import YOLO
import cv2
import math
import cvzone
from sort import *

cap = cv2.VideoCapture('Resources/cars.mp4')
mask = cv2.imread('Resources/mask.png')

#tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limitsLine = [300,380,673,380]
listIds = [0]
totalCount = 0

model = YOLO('Weights/yolov10m.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img,mask)
    result = model(imgRegion, stream=True)

    list_detections = np.empty((0,5))

    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print (x1, y1, x2, y2)

            #Confidence
            conf = math.ceil((box.conf[0]*100))/100
            print (conf)

            #Numero de la clase
            cls = int(box.cls[0])

            if (classNames[cls] == 'car' or classNames[cls] == 'motorbike' or
            classNames[cls] == 'bus' or classNames[cls] == 'truck' or
            classNames[cls] == 'bicycle' or classNames[cls] == 'person' and conf > 0.3):

                #Cuadro que muestra clase y confidence:
                ##cvzone.putTextRect(img, f'{classNames[cls]} {conf}',(x1,y1-20), scale=1, thickness=1,offset=5)

                # Rectangulo detector, no trackeador:
                ##cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),3,)

                currentArray = np.array([x1,y1,x2,y2,conf])
                list_detections = np.vstack((list_detections,currentArray))


    results_tracker = tracker.update(list_detections)

    line = cv2.line(img, (limitsLine[0], limitsLine[1]),(limitsLine[2], limitsLine[3]), (0,0,255), thickness=5)


    for results in results_tracker:
        x1,y1,x2,y2,Id = results
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print (results)
        w, h = x2-x1,y2-y1
        cvzone.cornerRect(img,(x1,y1,w,h),l=9, rt=2, colorR=(255,0,0))
        cvzone.putTextRect(img, f'{int(Id)}', (x1, y1 - 20), scale=1, thickness=1, offset=3)

        cx,cy = x1+ w//2, y1+h//2
        cv2.circle(img,(cx,cy),10,(255,0,255),cv2.FILLED)


        if limitsLine[0]<cx<limitsLine[2] and limitsLine[1]-20<cy<limitsLine[3]+20:
            if listIds.count(Id) == 0:
                totalCount += 1
                listIds.append(Id)
                line = cv2.line(img, (limitsLine[0], limitsLine[1]), (limitsLine[2], limitsLine[3]), (0, 255, 0), thickness=5)

    cvzone.putTextRect(img, f'Total Count:{totalCount}', (50,50) , colorR=(255,0,0))

    cv2.imshow('image',img)
    #cv2.imshow('region', imgRegion)
    cv2.waitKey(1)