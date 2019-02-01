import math

import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from ..dataset import get_digit_bbox


@pytest.fixture
def square():
    im = np.zeros((8, 8), dtype=np.uint8)
    im[2:-2, 2:-2] = 2
    return im


@pytest.fixture
def triangle():
    im = np.zeros((8, 8), dtype=np.uint8)
    for i in range(2, 6):
        start, end = 4 - math.ceil(i / 2), 4 + math.floor(i / 2)
        for j in range(start, end):
            im[i, j] = 1
    return im


@pytest.fixture
def two_regions():
    im = np.zeros((8, 8), dtype=np.uint8)
    im[3:5, 2:4] = 1
    im[1:6, 6:7] = 128
    return im


def test_get_digit_bbox1(square):
    assert_almost_equal(get_digit_bbox(square), np.array([2, 2, 5, 5]))


def test_get_digit_bbox2(triangle):
    assert_almost_equal(get_digit_bbox(triangle), np.array([1, 2, 5, 5]))


def test_get_digit_bbox3(two_regions):
    assert_almost_equal(get_digit_bbox(two_regions), np.array([2, 1, 6, 5]))


def test_get_digit_bbox4():
    assert_almost_equal(get_digit_bbox(np.load('digit3.npy')), np.array([4, 5, 23, 24]))
