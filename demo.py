from streamlit_webrtc import WebRtcMode, webrtc_streamer
from turn import get_ice_servers
import streamlit as st
import cv2
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import math
import os
from tensorflow.keras.applications.resnet50 import preprocess_input
import time
import av
import sys

detector = HandDetector(maxHands=1)

# Define the number of frames to use for prediction
num_frames = 5
# Initialize a list to store the frames
frame_buffer = []

# Load the trained model & set the input shape parameters for model
#model_path = sys.path.append('Model/model_resnet50_100_landmark.keras')
model = load_model('Model/model_resnet50_100_landmark.keras')
offset = 20
imgSize = 300
imgSize_to_model = 100
final_predict = ''
letters = {'0':'A', '1':'B', '2':'C', '3':'D', '4':'E', '5':'F', '6':'G', '7':'H', '8':'I', '9':'K', '10':'L', '11':'M', '12':'N',
           '13':'O', '14':'P', '15':'Q', '16':'R', '17':'S', '18':'T', '19':'U', '20':'V', '21':'W', '22':'X', '23':'Y'}

### App part
st.title("Sign language interpreter")
#frame_placeholder = st.empty()

def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = img[y-offset : y+h+offset, x-offset : x+w+offset]
        aspectRatio = h/w
        try:
            if aspectRatio > 1:
                k = imgSize/h
                wCal = math.ceil(k*w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize-wCal)/2)
                imgWhite[:, wGap:wCal+wGap] = imgResize
                # Preprocess the image
                image_resized = cv2.resize(imgWhite, (imgSize_to_model, imgSize_to_model))
                image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                image_pre = preprocess_input(np.expand_dims(image_rgb, axis=0))  # Add batch dimension & Preprocess the input according to ResNet50 requirements
                result = letters[str(int(np.argmax(model.predict(image_pre), axis=1)))]
                # Add frame to the buffer
                frame_buffer.append(result)
                # Keep only the last `num_frames` frames in the buffer
                if len(frame_buffer) > num_frames:
                    frame_buffer = frame_buffer[-num_frames:]
                # Once we have enough frames, make predictions
                if len(frame_buffer) == num_frames:
                    # Create a batch of frames
                    final_predict = pd.DataFrame(frame_buffer).reset_index(drop=True).value_counts().keys()[0][0]
            else:
                k = imgSize/w
                hCal = math.ceil(k*h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize-hCal)/2)
                imgWhite[hGap:hCal+hGap, :] = imgResize
                #prediction, index = classifier.getPrediction(imgWhite, draw=False)
                image_resized = cv2.resize(imgWhite, (imgSize_to_model, imgSize_to_model))
                image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                image_pre = preprocess_input(np.expand_dims(image_rgb, axis=0))  # Add batch dimension & Preprocess the input according to ResNet50 requirements
                result = letters[str(int(np.argmax(model.predict(image_pre), axis=1)))]
                # Add frame to the buffer
                frame_buffer.append(result)
                # Keep only the last `num_frames` frames in the buffer
                if len(frame_buffer) > num_frames:
                    frame_buffer = frame_buffer[-num_frames:]
                # Once we have enough frames, make predictions
                if len(frame_buffer) == num_frames:
                    final_predict = pd.DataFrame(frame_buffer).reset_index(drop=True).value_counts().keys()[0][0]
        except:
            frame_pr = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
            return av.VideoFrame.from_ndarray(frame_pr)
        cv2.rectangle(imgOutput, (x-offset, y-offset-50), (x-offset+90, y-offset), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, final_predict, (x, y-27), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255,255,255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (255, 0, 255), 4)
    frame_pr = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
    return av.VideoFrame.from_ndarray(frame_pr) #, format="bgr24"

webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": get_ice_servers(),
        "iceTransportPolicy": "relay",
    },
    video_frame_callback=callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
