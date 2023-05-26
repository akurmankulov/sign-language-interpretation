import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model  # TensorFlow is required for Keras to work

st.set_page_config(
    page_title="SLB Signs",
    page_icon=":auto:"#"👋",
    )

st.title("Shall we try to do it live?!")

import urllib.request
urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task', 'gesture_recognizer.task')

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an GestureRecognizer object.
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

#read image from webcam
picture = st.camera_input("Let's get your webcam in action and grab a picture...")
if picture:
     # Preprocess
    uploaded_image = Image.open(picture)
    image_np = np.array(uploaded_image)

    # Convert the image to RGB format if needed
    if image_np.shape[2] == 4:  # Check if image has an alpha channel
        image_np = image_np[:, :, :3]  # Remove alpha channel

    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)  # Convert to RGB format

    # Save the image to a temporary file
    temp_file = "temp.jpg"
    cv2.imwrite(temp_file, image_rgb)

    # Read the image using Mediapipe's imread function
    image_mp = mp.imread(temp_file)
    
    # STEP 4: Recognize gestures in the input image.
    recognition_result = recognizer.recognize(image_mp)
    top_gesture = recognition_result.gestures[0][0]
        
    # Print prediction and confidence score
    st.write("Letter:", top_gesture)
