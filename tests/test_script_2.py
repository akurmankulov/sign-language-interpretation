from PIL import Image
import numpy as np
import os

#this is to read file from main folder
import sys
sys.path.append('../')

from aslr import ASLRecognition

def test_read_array_shape(path):
    """
    Test the shape of the image array loaded by ASLRecognition for multiple images.

    Parameters:
    path (str): The path to the image.

    Raises:
    AssertionError: If the shape of the image array loaded by ASLRecognition
                    does not match the expected shape.

    """
    _a=ASLRecognition()
    _a.read_array(np.array(Image.open(path))) #converting the image to array
    assert _a.__dict__['array'].shape[0] == np.array(Image.open(path)).shape[0]
