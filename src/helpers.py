import numpy as np


def area(bboxes):
    widths = np.maximum(0, bboxes[:, 2] - bboxes[:, 0] + 1)
    heights = np.maximum(0, bboxes[:, 3] - bboxes[:, 1] + 1)
    areas = widths * heights
    return areas


def intersection(bboxes_a, bboxes_b):
    intsec_area_left = np.maximum(bboxes_a[:, 0], bboxes_b[:, 0])[..., None]
    intsec_area_right = np.minimum(bboxes_a[:, 2], bboxes_b[:, 2])[..., None]
    intsec_area_top = np.maximum(bboxes_a[:, 1], bboxes_b[:, 1])[..., None]
    intsec_area_bottom = np.minimum(bboxes_a[:, 3], bboxes_b[:, 3])[..., None]
    return np.concatenate((intsec_area_left, intsec_area_top, intsec_area_right, intsec_area_bottom), axis=1)


def iou(bboxes_a, bboxes_b):
    i = area(intersection(bboxes_a, bboxes_b))
    u = area(bboxes_a) + area(bboxes_b) - i
    return i / (u + 1e-7)
