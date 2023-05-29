import cv2
from cvzone.HandTrackingModule import HandDetector
#from cvzone.ClassificationModule import Classifier
from tensorflow.keras.models import load_model
import numpy as np
import math
import time
import os
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array  # also load_img
import tensorflow as tf

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
#classifier = Classifier("Model/model_1.h5", "Model/labels.txt")

model = load_model(os.path.join("models","model_resnet50_100_landmark"))
offset = 20
imgSize = 300
imgSize_to_model = 100

# with open('Model/labels.txt') as f:
#     labels = [i.split()[1] for i in f.readlines()]
letters = {'0':'A', '1':'B', '2':'C', '3':'D', '4':'E', '5':'F', '6':'G', '7':'H', '8':'I', '9':'K', '10':'L', '11':'M', '12':'N',
           '13':'O', '14':'P', '15':'Q', '16':'R', '17':'S', '18':'T', '19':'U', '20':'V', '21':'W', '22':'X', '23':'Y'}


while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = img[y-offset : y+h+offset, x-offset : x+w+offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h/w
        try:
            if aspectRatio > 1:
                k = imgSize/h
                wCal = math.ceil(k*w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize-wCal)/2)
                imgWhite[:, wGap:wCal+wGap] = imgResize
                #prediction, index = classifier.getPrediction(imgWhite, draw=False)
                # Preprocess the image
                image_resized = cv2.resize(imgWhite, (imgSize_to_model, imgSize_to_model))
                image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                image_pre = preprocess_input(np.expand_dims(image_rgb, axis=0))  # Add batch dimension & Preprocess the input according to ResNet50 requirements
                index = int(np.argmax(model.predict(image_pre), axis=1))

            else:
                k = imgSize/w
                hCal = math.ceil(k*h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize-hCal)/2)
                imgWhite[hGap:hCal+hGap, :] = imgResize
                #prediction, index = classifier.getPrediction(imgWhite, draw=False)
                image_resized = cv2.resize(imgWhite, (imgSize_to_model, imgSize_to_model))
                image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                image_pre = preprocess_input(np.expand_dims(image_rgb, axis=0))  # Add batch dimension & Preprocess the input according to ResNet50 requirements
                index = int(np.argmax(model.predict(image_pre), axis=1))
        except:
            continue

        cv2.rectangle(imgOutput, (x-offset, y-offset-50), (x-offset+90, y-offset), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, letters[str(index)], (x, y-27), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255,255,255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (255, 0, 255), 4)
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
    ### if you neeed to interrupt, press q
    if key == ord("q"):
        break
    
# Close all windows
cv2.destroyAllWindows()
    
