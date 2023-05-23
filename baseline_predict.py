import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPool2D
from tensorflow.keras.backend import expand_dims
import string

##################################################
# Usage:
# from baseline_predict import baseline
# model = baseline()
# model.predict(image)
#################################################


class  baseline:

    def __init__(self):
        self.letters_ = [letter for letter in string.ascii_lowercase]
        self.model_ = load_model('./base_line_model/base_cnn_model')

    def predict(self,file):
        """
        This function need to input a 28x28 size text file as a image. Convert it first if not in thie format
        """
        img = pd.read_csv(file,header=None)/255
        X = expand_dims(np.reshape(img,(28,28,1)),axis = 0)
        y = np.argmax(self.model_.predict(X))
        return self.letters_[y]
