import numpy as np
# import pandas as pd
import cv2
import string
from PIL import Image
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import  cross_validate, train_test_split
# from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential,load_model
# from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Rescaling,Dropout,BatchNormalization,RandomZoom, RandomRotation,RandomTranslation
from tensorflow.keras.backend import expand_dims
# from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
# from tensorflow.keras.preprocessing import image_dataset_from_directory

##################################################
# Usage:
# from aslr import ASLRecognition
# model = ASLRecognition()
# model.loadmodel('model_path') where model_path is the location of model
# model.predict(image) the image should be an numpy array with np.uint8 as dtype
#################################################


class  ASLRecognition:
    """
    To detect the American Sign Languate letters. Currently only 24 letters can be detected, exluding 'J' and 'Z" which are motions.
    The input should be color images with a width:height ratio 1:1
    """

    def __init__(self):
        self.letters_ = string.ascii_uppercase.replace('J','').replace('Z','')

    def loadmodel(self,model,threshold = 0.5):
        """
        A full path of the model is neeed, or the model may be from an url
        """
        self.model_ = load_model(model)
        self.threshold_ = threshold
        return self

    def input_preprocessing(self, image, color_mode = 'rgb', img_size = (300,300), rescale = None):
        """
        The input image format should be a numpy array, and the rescale is the scaler used to divide the image for normalization
        """
        # if color_mode != 'rgb':
        #     img = np.array(Image.fromarray(image).convert('RGB'))
        # else:
        #     img = np.array(Image.fromarray(image))

        img = np.array(image)

        if img_size != img.shape[:2]:
            img = cv2.resize(img,img_size)

        if rescale:
            img  = img*1.0/rescale

        return expand_dims(img,axis = 0)

    def predict(self, image, color_mode = 'rgb', img_size = (300,300), rescale = None):
        X = self.input_preprocessing(image,color_mode=color_mode,img_size=img_size,rescale=rescale)
        pred = self.model_.predict(X)
        prob = np.max(pred)
        y = np.argmax(pred)

        if prob > self.threshold_:
            return self.letters_[y],prob
        return  None

    def build_model(self):
        pass

    def train(self):
        pass
