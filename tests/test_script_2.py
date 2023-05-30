import pytest
from PIL import Image
import numpy as np
import os

#this is to read file from main folder
import sys
sys.path.append('../')

from aslr import ASLRecognition

# it is possible to have a list of values and fixture will automatically run for each
@pytest.fixture(params=["tests/test_image_A.jpg", "tests/test_image_F.jpg", "tests/test_image_U.jpg"])
def path(request):
    return request.param

def test_many_images_np_shape(path):
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
