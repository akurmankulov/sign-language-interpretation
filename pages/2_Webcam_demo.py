import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
sys.path.append('../')
from model_cnn02 import ModelCNN02

st.set_page_config(
    page_title="SLB Signs",
    page_icon=":selfie:"#"👋",
    )

st.sidebar.success("Select another demo")

st.title("Shall we try to do it live?!")

#read image from webcam
picture = st.camera_input("Let's get your webcam in action and grab a picture...")
if picture:
    #preprocess
    uploaded_image = Image.open(picture)
    res = cv2.resize(np.array(uploaded_image), dsize=(56, 56), interpolation=cv2.INTER_CUBIC)
    grayscale_image = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
    reshaped_image = np.reshape(grayscale_image, (56, 56, 1))

    #get the prediction
    model=ModelCNN02()
    answer = model.predict(reshaped_image)

    st.write("the sign means", answer)
