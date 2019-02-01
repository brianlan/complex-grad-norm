import torch
import numpy as np
from numpy.testing import assert_almost_equal

from ..helpers import area, intersection, iou


def test_area():
    bboxes = torch.tensor([[1, 2, 4, 6],
                           [3, 0, 5, 4],
                           [0, 0, 0, 0],
                           [6, 6, 6, 6],
                           [5, 5, 4, 4],
                           [1, 8, 2, 8],
                           [1, 8, 2, 6],
                           [-3, -3, -1, -1],
                           [1, 2, 0, 0],
                           [291.0, 291.0, 0.0, 0.0]], dtype=torch.float)
    areas = area(bboxes)
    assert_almost_equal(areas.numpy(), np.array([20, 15, 1, 1, 0, 2, 0, 9, 0, 0]))


def test_intersection():
    bboxes_a = torch.tensor([[1, 2, 4, 6],
                             [3, 0, 5, 4],
                             [1, 2, 4, 6],
                             [1, 2, 4, 6]], dtype=torch.float)
    bboxes_b = torch.tensor([[3, 0, 5, 4],
                             [1, 2, 4, 6],
                             [1, 8, 2, 8],
                             [-1, -1, 2, 2]], dtype=torch.float)
    intersections = intersection(bboxes_a, bboxes_b)
    assert_almost_equal(intersections.numpy(), np.array([[3, 2, 4, 4],
                                                         [3, 2, 4, 4],
                                                         [1, 8, 2, 6],
                                                         [1, 2, 2, 2]]))


def test_iou():
    bboxes_a = torch.tensor([[1, 2, 4, 6],
                             [3, 0, 5, 4],
                             [1, 8, 2, 8],
                             [2, 5, 2, 5],
                             [1, 2, 4, 6],
                             [1, 8, 2, 6],
                             [-1, -1, 2, 2]], dtype=torch.float)
    bboxes_b = torch.tensor([[3, 0, 5, 4],
                             [1, 2, 4, 6],
                             [1, 2, 4, 6],
                             [1, 2, 4, 6],
                             [0, 4, 1, 5],
                             [1, 2, 4, 6],
                             [1, 2, 4, 6]], dtype=torch.float)
    ious = iou(bboxes_a, bboxes_b)
    assert_almost_equal(ious.numpy(), np.array([6 / 29, 6 / 29, 0, 1 / 20, 2 / 22, 0, 2 / 34]))
