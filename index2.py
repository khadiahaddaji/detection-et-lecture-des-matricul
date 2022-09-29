import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr

def objectDetector(img):
    yolo = cv2.dnn.readNet("model.weights","darknet-yolov3.cfg")
    classes = []

    with open("classes.names", "r") as file:
        classes = [line.strip() for line in file.readlines()]
    layer_names = yolo.getLayerNames()
    output_layers = [layer_names[i-1] for i in yolo.getUnconnectedOutLayers()]
    colorRed = (0,0,255)
    colorGreen = (0,255,0)
    colorWhite = (255,255,255)

    height, width, channels = img.shape

    # # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    yolo.setInput(blob)
    outputs = yolo.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            #cv2.rectangle(img, (x+5, y), (x + w+5, y + h+5), colorGreen, 3)
            #cv2.putText(img, label, (x, y - 30), cv2.FONT_HERSHEY_PLAIN, 3, colorWhite, 2)
            cut=img[y-10:y+h+5,x-15:x+w+5]
            plt.imshow(cv2.cvtColor(cut, cv2.COLOR_BGR2RGB))
            reader = easyocr.Reader(['en'])
            result = reader.readtext(cut,paragraph="False")
            text = result[0][1]
            cv2.putText(img, text, (x, y - 30), cv2.FONT_HERSHEY_PLAIN, 1, colorWhite, 2)
           
    return img



    

input_image = cv2.imread("cars101.png")
image = objectDetector(input_image)
height, width = image.shape[:2]
resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

fig = plt.gcf()
fig.set_size_inches(18, 10)
plt.axis("off")
plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
plt.show()