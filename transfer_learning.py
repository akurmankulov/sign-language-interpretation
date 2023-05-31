import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPool2D
from tensorflow.keras.backend import expand_dims
import string
from PIL import Image

##################################################
# Usage:
# from baseline_predict import baseline
# model = baseline()
# model.predict(image)
#################################################


class  Transfer_Model:

    def __init__(self, input_d1=28, input_d2=28, input_chan=1, output_d1=32, output_d2=32, output_chan=3):
        # Get the letters 
        self.letters_ = [letter for letter in string.ascii_lowercase]
        
        # Retrieve the trained model
        self.model_ = load_model("transfer_models/model_vgg16_v1")
        
        # Input image dimensions (currently training data)
        self.input_d1 = input_d1
        self.input_d2 = input_d2
        self.input_chan = input_chan
        
        # output image dimensions
        self.output_d1 = output_d1
        self.output_d2 = output_d2
        self.output_chan = output_chan
        

    def predict(self,file):
        """
        This function need to input a 28x28 size text file as a image. Convert it first if not in thie format
        """
        # Scale the input images for deep learning
        img = pd.read_csv(file,header=None)/255
        X_gray = np.array(img).reshape(-1, self.input_d1, self.input_d2)

        new_image = np.vstack((np.hstack((X_gray[0], np.ones((self.input_d1,self.output_d2-self.input_d2))*255)),
                               np.ones((self.output_d1-self.input_d1, self.output_d2))*255))
        
        new_image = Image.fromarray(new_image).convert('RGB')
        X_rgb = np.expand_dims(np.array(new_image), axis=0)
        y_pred = np.argmax(self.model_.predict(X_rgb))
        
        return self.letters_[y_pred]