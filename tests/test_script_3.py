from PIL import Image
import numpy as np
import os

#this is to read file from main folder
import sys
sys.path.append('../')

from aslr import ASLRecognition

def test_resize(path):
    _a = ASLRecognition()
    _a.read_image(path)
    _a.preprocessing(img_size = (100,100))
    assert np.array(_a.__dict__['original_image']).shape == np.array(Image.open(path)).shape
    assert np.array(_a.__dict__['resize']).shape[0]/np.array(Image.open(path)).shape[0] < 1
    assert np.array(_a.__dict__['resize']).shape[1]/np.array(Image.open(path)).shape[1] < 1

def test_rescale(path):
    _a = ASLRecognition()
    _a.read_image(path)
    _a.preprocessing(img_size = (100,100), rescale = True)
    assert np.array(_a.__dict__['array']).max() <= 1
