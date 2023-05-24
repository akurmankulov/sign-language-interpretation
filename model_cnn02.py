import numpy as np
import pandas as pd
import string
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import  cross_validate, train_test_split
# from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Rescaling,Dropout,BatchNormalization,RandomZoom, RandomRotation,RandomTranslation
from tensorflow.keras.backend import expand_dims
# from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
from PIL import Image

##################################################
# Usage:
# from model_cnn01 import ModelCNN01
# model = ModelCNN02()
# model.predict(image)
#################################################


class  ModelCNN02:

    def __init__(self):
        self.letters_ = [letter for letter in string.ascii_lowercase]
        self.model_ = load_model('./cnn_model/cnn_model_02')

    def predict(self,img,threshold = 0.4):
        """
        This function need to input a 56x56 size image. Convert it first if not in thie format
        """
        if img.shape == (56,56):
            X = expand_dims(np.reshape(img,(56,56,1)),axis = 0)
        elif img.shape ==(56,56,1):
            X = expand_dims(img,axis = 0)
        else:
            X = None

        if X == None:
            res = "The incorrect input shape"
        elif np.max(self.model_.predict(X)) < threshold:
            res = "Not a sign language"
        else:
            res = self.letters_[np.argmax(self.model_.predict(X))]

        return res
