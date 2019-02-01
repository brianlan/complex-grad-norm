import torch
import torch.nn.functional as F

from .helpers import iou


def classification_accuracy(pred_logits, gt_classes):
    scores = F.softmax(pred_logits, dim=1)
    pred_classes = torch.argmax(scores, dim=1)
    correct = (pred_classes == gt_classes).sum().float()
    accuracy = correct / len(gt_classes)
    return accuracy


def counter_accuracy(pred_counts, gt_counts):
    correct = (pred_counts[:, 0].round() == gt_counts).sum().float()
    accuracy = correct / len(gt_counts)
    return accuracy


def bbox_iou(pred_bboxes, gt_bboxes):
    ious = iou(pred_bboxes, gt_bboxes)
    return ious.mean()
