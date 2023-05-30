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
# model.loadmodel('model_path') where model_path is the location of model, and threshold for the probability can be specified. Default is 0.5
# model.read_image() or model.read_array() to read the input as an jpg file or an array
# model.preprocessing().predict(image) to do proprocessing and then predict. The input image size should
# be specified in the preprocessing if not (300,300).
#################################################


class  ASLRecognition:
    """
    To detect the American Sign Languate letters. Currently only 24 letters can be detected, exluding 'J' and 'Z" which are motions.
    The input should be color images with a width:height ratio 1:1
    """

    def __init__(self):
        # self.__dict__ = {'letters':string.ascii_uppercase.replace('J','').replace('Z','')}
        self.__dict__ = {}
        self.labels_ = list(string.ascii_uppercase.replace('J','').replace('Z',''))

    def loadmodel(self,model,threshold = 0.5):
        """
        A full path of the model is neeed, or the model may be from an url
        """
        self.model_ = load_model(model)
        self.threshold_ = threshold
        self.__dict__['model'] = self.model_
        return self

    def read_image(self,image):
        """ If input is an image, then read the image to convert it to numpy array"""
        self.__dict__['original_image'] = Image.open(image)
        self.__dict__['RGB_image'] = self.__dict__['original_image'].convert('RGB')
        self.image_ = np.array(self.__dict__['RGB_image'])
        self.__dict__['array'] = self.image_
        self.__dict__['resize'] = None
        return self

    def read_array(self,image_array):
        self.__dict__['original_image'] = None
        self.__dict__['RGB_image'] = None
        self.__dict__['resize'] = None
        self.image_ = image_array
        self.__dict__['array'] = self.image_
        return self

    def preprocessing(self, img_size = (300,300), rescale = None):
        """
        The input image format should be a numpy array, and the rescale is the scaler used to divide the image for normalization
        """

        if img_size != self.image_.shape[:2]:
            self.image_ = self.image_.resize(img_size)
            self.__dict__['resize'] = self.image_

        if rescale:
            self.image_ = self.image_*1.0/255

        self.image_  = expand_dims(self.image_,axis = 0)

        return self

    def predict(self):
        pred = self.model_.predict(self.image_)
        prob = np.max(pred)
        y = np.argmax(pred)

        if prob > self.threshold_:
            return self.labels_[y],prob
        return  None

    def build_model(self):
        pass

    def train(self):
        pass
