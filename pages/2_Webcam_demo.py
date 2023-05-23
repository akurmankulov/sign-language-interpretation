import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
sys.path.append('../')
from baseline_predict import Baseline

# st.set_page_config(
#     page_title="SLB Signs",
#     page_icon=":selfie:"#"👋",
#     )

# st.sidebar.success("Select another demo")

# st.title("Shall we try to do it live?!")

# #read image from webcam
# picture = st.camera_input("Let's get your webcam in action and grab a picture...")
# if picture:
#     #preprocess
#     grayscale_image = cv2.cvtColor(picture, cv2.COLOR_RGB2GRAY)
#     reshaped_image = np.reshape(grayscale_image, (28, 28, 1))

#     #get the prediction
#     model=Baseline()
#     answer = model.predict(reshaped_image)

st.write("the sign means")#, answer)
