import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
from sklearn.metrics import accuracy_score
from sklearn.model_selection import  cross_validate, train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Rescaling,Dropout,BatchNormalization,RandomZoom, RandomRotation,RandomTranslation
from tensorflow.keras.backend import expand_dims
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint

##################################################
# Usage:
# from model_cnn01 import ModelCNN01
# model = ModelCNN01()
# model.predict(image)
#################################################


class  ModelCNN01:

    def __init__(self):
        self.letters_ = [letter for letter in string.ascii_lowercase]
        self.model_ = load_model('./cnn_model/cnn_model_01')
        self.threshold_ = 0.4

    def predict(self,img):
        """
        This function need to input a 28x28 size image. Convert it first if not in thie format
        """
        if img.shape == (28,28):
            X = expand_dims(np.reshape(img,(28,28,1)),axis = 0)
        else:
            X = expand_dims(img,axis = 0)

        y = np.argmax(self.model_.predict(X))
        return self.letters_[y]
