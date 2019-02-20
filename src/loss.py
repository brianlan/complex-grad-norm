import torch
import torch.nn as nn
import torch.nn.functional as F


def class_loss(pred, gt_classes):
    return F.cross_entropy(pred, gt_classes, reduction='mean').unsqueeze(0)


def bbox_loss(pred, gt_bboxes):
    return F.smooth_l1_loss(pred, gt_bboxes, reduction='mean').unsqueeze(0)


def counter_loss(pred, gt_counts):
    return F.mse_loss(pred, gt_counts, reduction='mean').unsqueeze(0)
