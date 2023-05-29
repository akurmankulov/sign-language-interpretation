import streamlit as st
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from tensorflow.keras.models import load_model
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
#classifier = Classifier("Model/model_1.h5", "Model/labels.txt")

model = load_model("./base_line_model/base_cnn_model")
offset = 20
imgSize = 300

with open('labels_4.txt') as f:
    labels = [i.split()[1] for i in f.readlines()]

### App part
st.title("Sign language interpreter")
frame_placeholder = st.empty()
stop_button_pressed = st.button("Stop")

while cap.isOpened() and not stop_button_pressed:
    success, img = cap.read()
    if not success:
        st.write('The video capture has ended')
        break

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
                index = int(np.argmax(model.predict(imgWhite.reshape(-1,300,300,3)), axis=1))

            else:
                k = imgSize/w
                hCal = math.ceil(k*h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize-hCal)/2)
                imgWhite[hGap:hCal+hGap, :] = imgResize
                #prediction, index = classifier.getPrediction(imgWhite, draw=False)
                index = int(np.argmax(model.predict(imgWhite.reshape(-1,300,300,3)), axis=1))
        except:
            continue

        cv2.rectangle(imgOutput, (x-offset, y-offset-50), (x-offset+90, y-offset), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y-27), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255,255,255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (255, 0, 255), 4)

    #cv2.imshow("Image", imgOutput)
    frame = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame)
    key = cv2.waitKey(1)
    if key == ord("q") or stop_button_pressed:
        break

cap.release()
