import pytest
from test_script_1 import *
from test_script_2 import *
from test_script_3 import *
from test_script_4 import *


#you can specify a test variable aka fixture
@pytest.fixture
def path1():
    return "tests/test_image_A.jpg"

@pytest.fixture
def path2():
    return "tests/test_image_G.jpg"
    return "tests/test_image_U.jpg"

@pytest.fixture
def path3():
    return "tests/test_image_F.jpg"

# it is possible to have a list of values and fixture will automatically run for each
@pytest.fixture(params=["tests/test_image_A.jpg",
                        "tests/test_image_F.jpg",
                        "tests/test_image_T.jpg",
                        "tests/test_image_G.jpg"])
def path(request):
    return request.param
