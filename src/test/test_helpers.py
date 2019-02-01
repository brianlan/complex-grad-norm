import numpy as np
from numpy.testing import assert_almost_equal

from ..helpers import area, intersection, iou


def test_area():
    bboxes = np.array([[1, 2, 4, 6],
                       [3, 0, 5, 4],
                       [0, 0, 0, 0],
                       [6, 6, 6, 6],
                       [5, 5, 4, 4],
                       [1, 8, 2, 8],
                       [1, 8, 2, 6],
                       [-3, -3, -1, -1],
                       [1, 2, 0, 0],
                       [291.0, 291.0, 0.0, 0.0]])
    areas = area(bboxes)
    assert_almost_equal(areas, np.array([20, 15, 1, 1, 0, 2, 0, 9, 0, 0]))


def test_intersection():
    bboxes_a = np.array([[1, 2, 4, 6],
                         [3, 0, 5, 4],
                         [1, 2, 4, 6],
                         [1, 2, 4, 6]])
    bboxes_b = np.array([[3, 0, 5, 4],
                         [1, 2, 4, 6],
                         [1, 8, 2, 8],
                         [-1, -1, 2, 2]])
    intersections = intersection(bboxes_a, bboxes_b)
    assert_almost_equal(intersections, np.array([[3, 2, 4, 4],
                                                 [3, 2, 4, 4],
                                                 [1, 8, 2, 6],
                                                 [1, 2, 2, 2]]))


def test_iou():
    bboxes_a = np.array([[1, 2, 4, 6],
                         [3, 0, 5, 4],
                         [1, 8, 2, 8],
                         [2, 5, 2, 5],
                         [1, 2, 4, 6],
                         [1, 8, 2, 6],
                         [-1, -1, 2, 2]])
    bboxes_b = np.array([[3, 0, 5, 4],
                         [1, 2, 4, 6],
                         [1, 2, 4, 6],
                         [1, 2, 4, 6],
                         [0, 4, 1, 5],
                         [1, 2, 4, 6],
                         [1, 2, 4, 6]])
    ious = iou(bboxes_a, bboxes_b)
    assert_almost_equal(ious, np.array([6 / 29, 6 / 29, 0, 1 / 20, 2 / 22, 0, 2 / 34]))
