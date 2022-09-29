from operator import index
import cv2
import numpy as np

net=cv2.dnn.readNetFromDarknet('darknet-yolov3.cfg','model.weights')

DEFAULT_CONFIANCE  =0.5
THRESHOLD=0.4
with open('classes.names','r') as f:
    classes = [line.strip()for line in f.readlines()]

#cap=cv2.VideoCapture('video13.mp4')


while True :
    #success,image=cap.read()
    image=cv2.imread('55.jpg') 
    image=cv2.resize(image,(900,600))
    height , width,_ = image.shape

    blob = cv2.dnn.blobFromImage ( image , 1/255 , ( 416 , 416 ) , ( 0,0,0 ) , swapRB =True , crop = False )
    net.setInput ( blob )
    last_layer=net.getUnconnectedOutLayersNames()
    layerOutputs=net.forward(last_layer)
    boxes= [ ]
    confidences= [ ]
    classIDs =[]

      
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > DEFAULT_CONFIANCE:
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, W, H) = box.astype("int")
                x = int(centerX - (W / 2))
                y = int(centerY - (H / 2))
                boxes.append([x, y, int(W), int(H)])
                confidences.append(float(confidence))
                classIDs.append(classID)
        
   
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, DEFAULT_CONFIANCE, THRESHOLD)
    print(index)
   
    COLORS = np.random.uniform(0,255,size=(len(boxes), 3))

   
    if len(indexes) > 0:
       
        for i in indexes.flatten():
          
            (x, y, w, h) = boxes[i]
            color = COLORS[i]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            

    cv2.imshow('Image', image)
    cv2.waitKey(4)

cv2.destroyAllWindows()