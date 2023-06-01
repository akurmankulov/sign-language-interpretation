from streamlit_webrtc import WebRtcMode, webrtc_streamer, VideoProcessorBase
from turn import get_ice_servers
import streamlit as st
import cv2
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import math
from tensorflow.keras.applications.resnet50 import preprocess_input
import time
import av
import sys

sys.path.append('../')
model = load_model('../Model/model_resnet50_100_landmark.keras')
offset = 20
imgSize = 300
imgSize_to_model = 100
letters = {'0':'A', '1':'B', '2':'C', '3':'D', '4':'E', '5':'F', '6':'G', '7':'H', '8':'I', '9':'K', '10':'L', '11':'M', '12':'N',
           '13':'O', '14':'P', '15':'Q', '16':'R', '17':'S', '18':'T', '19':'U', '20':'V', '21':'W', '22':'X', '23':'Y'}

### App part
st.title("Sign language interpreter")
delete_button_pressed = st.button("Delete")
text_placeholder = st.empty()

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = HandDetector(maxHands=1)
        self.frame_buffer = []
        self.num_frames = 5
        self.final_predict = ''
        self.final_text = ''
        self.t = 0
        self.letters2display = []
        self.text_gen_time = 25

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        imgOutput = img.copy()
        hands, img = self.detector.findHands(img)
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
                    self.frame_buffer.append(result)
                    # Keep only the last `num_frames` frames in the buffer
                    if len(self.frame_buffer) > self.num_frames:
                        self.frame_buffer = self.frame_buffer[-self.num_frames:]
                    # Once we have enough frames, make predictions
                    if len(self.frame_buffer) == self.num_frames:
                        # Create a batch of frames
                        self.final_predict = pd.DataFrame(self.frame_buffer).reset_index(drop=True).value_counts().keys()[0][0]
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
                    self.frame_buffer.append(result)
                    # Keep only the last `num_frames` frames in the buffer
                    if len(self.frame_buffer) > self.num_frames:
                        self.frame_buffer = self.frame_buffer[-self.num_frames:]
                    # Once we have enough frames, make predictions
                    if len(self.frame_buffer) == self.num_frames:
                        self.final_predict = pd.DataFrame(self.frame_buffer).reset_index(drop=True).value_counts().keys()[0][0]
            except:
                #frame_pr = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
                return av.VideoFrame.from_ndarray(imgOutput, format="bgr24")
            cv2.rectangle(imgOutput, (x-offset, y-offset-50), (x-offset+90, y-offset), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, self.final_predict, (x, y-27), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255,255,255), 2)
            cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (255, 0, 255), 4)
            self.letters2display.append(self.final_predict)
            if len(self.letters2display) % self.text_gen_time==0:
                letter = pd.DataFrame(self.letters2display[-self.text_gen_time:] ).reset_index(drop=True).value_counts().keys()[0][0]
                st.session_state['final_text']+=letter
                text_placeholder.header(st.session_state['final_text'])
            self.t=0
        t+=1
        if t == 40:
            self.final_text += ' '
        if delete_button_pressed:
            self.final_text = self.final_text[:-1]
            text_placeholder.header(self.final_text)
            delete_button_pressed=False
        #frame_pr = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
        return av.VideoFrame.from_ndarray(imgOutput, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": get_ice_servers(),
        "iceTransportPolicy": "relay",
    },
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
