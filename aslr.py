import numpy as np
# import pandas as pd
# import cv2
import string
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential
import tensorflow.keras.layers as layers
from tensorflow.keras.backend import expand_dims
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
import tensorflow.keras.applications as pretrain_models

##################################################
# How to Use:
# from aslr import ASLRecognition
#
# model = ASLRecognition()
# model.loadmodel('model_path')             #where model_path is the location of model, and threshold for the
#                                           #probability can be specified. Default is 0.5
# model.read_image() or model.read_array()  #read the input as an jpg file or an array
# model.preprocessing().predict(image)      #proprocessing and then predict. The input image size should
#                                           #be specified in the preprocessing if not (300,300).
#################################################


class  ASLRecognition:
    """
    Class for detecting American Sign Language letters.
    Currently supports detection of 24 letters, excluding 'J' and 'Z' which represent motions.
    Input images should have a width-to-height ratio of 1:1.
    """

    def __init__(self):
        """
        Initialize ASLRecognition object. Labels list is created here.
        """
        self.__dict__ = {}
        self.__dict__['labels'] = list(string.ascii_uppercase.replace('J','').replace('Z',''))

    def loadmodel(self,model,threshold = 0.5):
        """
        Load a model for ASL recognition.

        Args:
            model (str): Full path or URL of the model.
            threshold (float, optional): Probability threshold for predictions. Default is 0.5.

        Returns:
            ASLRecognition: The current ASLRecognition object.
        """
        try:
            self.threshold_ = threshold
            self.__dict__['model'] = load_model(model)
            self.model_ = self.__dict__['model']
            self.model_trainable_ = False
            return self

        except ValueError:
            print("Error: cannot find or load the model")
            return None

    def read_image(self,image_file):
        """
        Read an input image and convert it to a NumPy array.

        Args:
            image (str): Path to the input image file.

        Returns:
            ASLRecognition: The current ASLRecognition object.
        """
        try:
            self.__dict__['original_image'] = Image.open(image_file)
            self.__dict__['RGB_image'] = self.__dict__['original_image'].convert('RGB')
            self.image_ = self.__dict__['RGB_image']
            self.__dict__['array'] = None
            self.__dict__['resize'] = None
            return self

        except FileNotFoundError:
            print("Error: File not found or corrupte")
            return None

    def read_array(self,image_array):
        """
        Read an input image as a NumPy array.

        Args:
            image_array (numpy.ndarray): Input image as a NumPy array.

        Returns:
            ASLRecognition: The current ASLRecognition object.
        """
        try:
            self.__dict__['original_image'] = Image.fromarray(image_array)
            self.image_ = self.__dict__['original_image']
            self.__dict__['RGB_image'] = None
            self.__dict__['resize'] = None
            self.__dict__['array'] = None
            return self
        except ValueError:
            print('Error: Unrecognized Image')
            return None

    def preprocessing(self, img_size = (300,300), rescale = False):#None):
        """
        Preprocess the input image.

        Args:
            img_size (tuple, optional): Size of the image after resizing. Default is (300, 300).
            rescale (float, optional): Scaling factor to normalize the image. Default None is no scaling.

        Returns:
            ASLRecognition: The current ASLRecognition object.
        """

        try:
            if img_size != np.array(self.image_).shape[:2]:
                    self.__dict__['resize'] = self.image_.resize(img_size)
                    self.array_ = np.array(self.__dict__['resize'])
            else:
                self.array_ = np.array(self.image_)
        except Exception:
                print("Resize failed!")
                return None


        if rescale:
            self.array_ = self.array_*1.0/255

        self.array_  = expand_dims(self.array_,axis = 0)

        return self

    def predict(self):
        """
        Make a prediction based on the preprocessed image.

        Returns:
            tuple: A tuple containing the predicted letter and its probability,
                   or None if the probability is below the threshold.
        """
        try:
            pred = self.model_.predict(self.array_)

        except Exception:
            print("Error: model prediction failed")
            return None

        prob = np.max(pred)
        y = np.argmax(pred)

        if prob > self.threshold_:
            return self.__dict__['labels'][y],prob
        return  None

    def initialize_InceptionV3_model(self,input_shape = (300,300,3),num_class = 24):
        """
        This is Jianfe's model using InceptionV3 as a pretrained model and with only 1 dense layer. The realtime
        test looks good.
        Args:
            input_shape: the input_shape. Default is (300,300,3)
            num_class: the number for multiclassification. Default is 24 for 24 static American Sign Language Letters.
        Returns:
            The compiled model.
        """
        base = pretrain_models.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(300,300,3))
        base.trainable = False

        self.model_ = Sequential()
        # Add an scaling layers to normalize the image
        self.model_.add(layers.Rescaling(1./255,input_shape = input_shape))

        ### Use the Pretrained ResNet50 Model
        self.model_.add(base)

        ### Flattening
        self.model_.add(layers.Flatten())

        ### First Fully Connected layer - "Fully Connected" is equivalent to saying "Dense"
        self.model_.add(layers.Dense(256,activation = 'relu'))

        ### Last layer - Classification Layer with 29 outputs
        self.model_.add(layers.Dense(num_class,activation = 'softmax'))

        ### Model compilation
        self.model_.compile(loss = 'categorical_crossentropy', optimizer = 'adam',metrics = ['accuracy'])

        self.model_trainable = True

        return self

    def initialize_ResNet50_model(self,image_width=100, image_height=100, no_chans=3, n_classes=24):
        """
        This is Yulian's model using ResNet50 as a pretrained model. The realtime test looks good.
        Args:
            input_shape: the input_shape. Default is (100,100,3)
            num_class: the number for multiclassification. Default is 24 for 24 static American Sign Language Letters.
        Returns:
            The compiled model.
        """
        #Initialising ResNet50
        basemodel_resnet = pretrain_models.resnet50.ResNet50(input_shape=(image_width,image_height,no_chans),include_top=False,weights='imagenet')

        #don't train existing weights for resnet50
        for layer in basemodel_resnet.layers:
            layer.trainable = False
        # basemodel_resnet.trainable = False

        # Alternative way to add the other necessary layers for pre-trained model
        self.model_ = Sequential()
        self.model_.add(basemodel_resnet)
        self.model_.add(layers.Flatten())
        self.model_.add(layers.Dense(512, activation='relu'))
        self.model_.add(layers.Dense(256, activation='relu'))
        self.model_.add(layers.Dense(128, activation='relu'))
        self.model_.add(layers.Dropout(0.2))
        self.model_.add(layers.Dense(units=n_classes, activation='softmax'))

        #Build and compile the model
        self.model_.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model_trainable_ = True
        return self

    def fit(self,X_train,y_train,X_val,y_val,epochs = 50,batch_size = 64,verbose = 0,callbacks = ['es']):
        if self.model_trainable:
            es = EarlyStopping(patience = 5, restore_best_weights = True)
            reduc_lr = ReduceLROnPlateau(patience = 3,verbose = verbose)
            self.model_.fit(X_train,y_train,validation_data = (X_val,y_val),callbacks=['es','reduc_lr'])
            return self
        return None
