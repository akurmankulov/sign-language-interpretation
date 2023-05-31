import streamlit as st
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import pandas as pd
import math
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf
import os

def load_model(source, model_path):
    """Load the model from Google Cloud Storage."""
    if source == 'GCP':
        model_url = f"gs://{model_path}"
        model = tf.keras.models.load_model(model_url)
    elif source == 'local':
        model = tf.keras.models.load_model(model_path)
    return model

def main():
    
    st.title("Sign Language Demo in Streamlit")
    
    # Google Cloud Storage information
    source = 'GCP'  # GCP or local
    model_name = 'model_resnet50_100_landmark.keras'
    
    if source == 'GCP':
        model_path = os.path.join("sign_language_demo", "models", model_name)
    else:
        model_path = os.path.join('.', 'models', model_name)
    
    # Load the model from gcs or local
    # model = load_model_from_gcs(bucket_name, model_path)
    model = load_model(source, model_path)
    
    # Create a VideoCapture object
    video_capture = cv2.VideoCapture(0)
  
    # Check if the video capture is successfully opened
    if not video_capture.isOpened():
        st.error("Failed to open the video capture.")
        return
    
    # Read the first frame
    _, frame = video_capture.read()
    detector = HandDetector(maxHands=1)
    
    # Display the frame using Streamlit
    st.image(frame, channels="BGR") 

    # Define the number of frames to use for prediction
    num_frames = 5

    # Initialize a list to store the frames
    frame_buffer = []

    # Load the trained model & set the input shape parameters for model
    offset = 20
    imgSize = 300
    imgSize_to_model = 100

    # Decode the prediction letters
    letters = {'0':'A', '1':'B', '2':'C', '3':'D', '4':'E', '5':'F', '6':'G', '7':'H', '8':'I', '9':'K', '10':'L', '11':'M', '12':'N',
            '13':'O', '14':'P', '15':'Q', '16':'R', '17':'S', '18':'T', '19':'U', '20':'V', '21':'W', '22':'X', '23':'Y'}

    while True:
        success, img = video_capture.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        
        # If the hands are detected
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
            imgCrop = img[y-offset : y+h+offset, x-offset : x+w+offset]

            final_predict = ' '
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
                        # print(f'final_predict is {final_predict}')          
            except:
                continue

            cv2.rectangle(imgOutput, (x-offset, y-offset-50), (x-offset+90, y-offset), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, final_predict, (x, y-27), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255,255,255), 2)
            cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (255, 0, 255), 4) 
            # cv2.imshow("ImageCrop", imgCrop)
            # cv2.imshow("ImageWhite", imgWhite)
            cv2.imshow("Image", imgOutput)
            # st.image(imgOutput)
            # Display the frame using Streamlit
            # st.image(imgOutput, channels="BGR")
            st.subheader(f"Prediction:{final_predict}")
                
        ### if you neeed to interrupt, press q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
     
    # Release the video capture object and close the window
    video_capture.release()   
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()

