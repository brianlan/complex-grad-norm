import torch.nn.functional as F


def classification_accuracy(pred_logits, gt_classes):
    scores = F.softmax(pred_logits, axis=1)
    pred_classes = torch.argmax(scores)
    correct = (pred_classes == gt_classes).sum()
    accuracy = correct / len(gt_classes)
    return accuracy


def counter_accuracy(pred_counts, gt_counts):
    correct = (pred_counts, gt_counts)
    accuracy = correct / len(gt_counts)
    return accuracy


def bbox_iou(pred_bboxes, gt_bboxes):
    intsec_area_left = max(self.left, other.left)
    intsec_area_right = min(self.right, other.right)
    intsec_area_top = max(self.top, other.top)
    intsec_area_bottom = min(self.bottom, other.bottom)
