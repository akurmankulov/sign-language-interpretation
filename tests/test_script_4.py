import pytest
from PIL import Image
import numpy as np
import os

#this is to read file from main folder
import sys
sys.path.append('../')

from aslr import ASLRecognition

@pytest.mark.parametrize('model_path , model_shape', [ ("tests/models/model_resnet50_100_landmark.keras", (None, 100, 100, 3)),
                                                ("tests/models/cnn_model_02", (None, 56, 56, 1)),
                                                ('tests/models/model_1.h5', (None, 300,300,3))] )
def test_loadmodel(model_path , model_shape):
    _a = ASLRecognition()
    _a.loadmodel(model_path)
    assert _a.__dict__['model'].name.find("seq") >= 0
    assert _a.__dict__['model'].input_shape == model_shape

def test_predict(path):
    _a = ASLRecognition()
    _a.loadmodel("tests/models/model_resnet50_100_landmark.keras", threshold = 0.4)
    _a.read_image(path)
    _a.preprocessing(img_size = (100,100), rescale = True)
    assert _a.predict()[0] == path[-5:-4]
