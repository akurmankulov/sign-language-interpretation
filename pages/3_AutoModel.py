import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model  # TensorFlow is required for Keras to work

st.set_page_config(
    page_title="SLB Signs",
    page_icon=":auto:"#"👋",
    )

st.sidebar.success("Select another demo")

st.title("Shall we try to do it live?!")

#get the prediction from loaded model
auto_model = load_model("pages/TM/TM_keras_Model.h5", compile=False)
# Load the labels
class_names = open("pages/TM/TM_labels.txt", "r").readlines()


#read image from webcam
picture = st.camera_input("Let's get your webcam in action and grab a picture...")
if picture:
    #preprocess
    uploaded_image = Image.open(picture)

    # resizing the image to be at least 224x224 and then cropping from the center
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    res = cv2.resize(np.array(uploaded_image), dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    grayscale_image = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
    reshaped_image = np.reshape(grayscale_image, (224, 224, 1))


    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = auto_model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    st.write("Letter:", class_name[2:], end="")
    st.write("Confidence Score:", confidence_score)
