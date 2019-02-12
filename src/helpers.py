import torch


def area(bboxes):
    nboxes = len(bboxes)
    zeros = torch.zeros(nboxes, dtype=bboxes.dtype, device=bboxes.device)
    widths = torch.max(bboxes[:, 2] - bboxes[:, 0] + 1, zeros)
    heights = torch.max(bboxes[:, 3] - bboxes[:, 1] + 1, zeros)
    areas = widths * heights
    return areas


def intersection(bboxes_a, bboxes_b):
    intsec_area_left = torch.max(bboxes_a[:, 0], bboxes_b[:, 0])[..., None]
    intsec_area_right = torch.min(bboxes_a[:, 2], bboxes_b[:, 2])[..., None]
    intsec_area_top = torch.max(bboxes_a[:, 1], bboxes_b[:, 1])[..., None]
    intsec_area_bottom = torch.min(bboxes_a[:, 3], bboxes_b[:, 3])[..., None]
    return torch.cat((intsec_area_left, intsec_area_top, intsec_area_right, intsec_area_bottom), dim=1)


def iou(bboxes_a, bboxes_b):
    i = area(intersection(bboxes_a, bboxes_b))
    u = area(bboxes_a) + area(bboxes_b) - i
    return i / (u + 1e-7)


def get_current_lr(optimizer):
    return min(g["lr"] for g in optimizer.param_groups)
