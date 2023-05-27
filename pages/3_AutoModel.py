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
#     st.write("type of uploaded image is:", type(image_np))
    
    # Load the input image from a numpy array.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
    
    # STEP 4: Recognize gestures in the input image.
    recognition_result = recognizer.recognize(mp_image)
    top_gesture = recognition_result.gestures[0][0].category_name
        
    # Set up dictionary to convert pre-defined gestures:
    dict ={"Unrecognized" : "Unrecognized",
          "Closed_Fist" : "Calm down",
           "Open_Palm" : "Let the magic begins",
           "Pointing_Up" : "it's leter D, remember?!",
           "Thumb_Down" : "Have to repeat the bootcamp",
           "Thumb_Up" : "Good job! True SLB spirit",
           "Victory" : "it's still letter V",
           "ILoveYou" : "rock'n'roll",
           "None" : "Try harder"
          }
    
    # Print prediction and confidence score
    st.write(dict[top_gesture])
