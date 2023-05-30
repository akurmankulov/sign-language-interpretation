import pytest
from PIL import Image
import numpy as np
import os

#this is to read file from main folder
import sys
sys.path.append('../')

from aslr import ASLRecognition

#you can specify a test variable aka fixture
@pytest.fixture
def path1():
    return "tests/test_image_A.jpg"

@pytest.fixture
def path2():
    return "tests/test_image_U.jpg"

# it is possible to have a list of values and fixture will automatically run for each
@pytest.fixture(params=["tests/test_image_A.jpg", "tests/test_image_F.jpg", "tests/test_image_U.jpg"])
def path(request):
    return request.param

# a direct variable creation also works of course
path_var="tests/test_image_F.jpg"

@pytest.mark.parametrize('input_path , expected_type', [ ("tests/test_image_A.jpg", str), ("tests/test_image_U.jpg", str)] )
def test_image_path(input_path, expected_type):
    """
    Test the type of the input path.

    Parameters:
    path (str): The path to the image.

    Raises:
    AssertionError: If the type of the path is not a string.

    """
    _a = ASLRecognition()
    _a.read_image(input_path)
    assert type(input_path) == expected_type

@pytest.mark.parametrize('input_path , shape', [ (path_var, (200, 200, 3)), ("tests/test_image_U.jpg", (300, 300, 3))] )
def test_image_orig_shape(input_path, shape):
    """
    Test the shape of the original image loaded by ASLRecognition.

    Parameters:
    path (str): The path to the image.

    Raises:
    AssertionError: If the shape of the original image loaded by ASLRecognition
                    does not match the shape of the image loaded using PIL.

    """
    _a = ASLRecognition()
    _a.read_image(input_path)
    assert np.array(_a.__dict__['original_image']).shape == shape

def test_image_np_shape(path1):
    """
    Test the shape of the image array loaded by ASLRecognition.

    Parameters:
    path (str): The path to the image.

    Raises:
    AssertionError: If the shape of the image array loaded by ASLRecognition
                    does not match the shape of the image loaded using PIL.

    """
    _a=ASLRecognition()
    _a.read_image(path1)
    assert _a.__dict__['array'].shape[0] == 300

def test_many_images_np_shape(path):
    """
    Test the shape of the image array loaded by ASLRecognition.

    Parameters:
    path (str): The path to the image.

    Raises:
    AssertionError: If the shape of the image array loaded by ASLRecognition
                    does not match the shape of the image loaded using PIL.

    """
    _a=ASLRecognition()
    _a.read_image(path)
    assert _a.__dict__['array'].shape[0] == np.array(Image.open(path)).shape[0]


# this is no longer needed with latest pytest versions
# if __name__ == "__main__":
    # print("Let the testing commence")
    # test_image_path(path)
    # test_image_orig_shape(path)
    # test_image_np_shape(path)
