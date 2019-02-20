import argparse
from pathlib import Path as P
import functools
import math

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from src.helpers import get_current_lr
from src.model import Net
from src.dataset import MNIST
from src.loss import class_loss, bbox_loss, counter_loss
from src.measure import classification_accuracy, counter_accuracy, bbox_iou
from src.logger import get_logger
from src.progress_visualizer import ProgressVisualizer
from src.gradnorm import SimpleGradNormalizer

parser = argparse.ArgumentParser("Training script")
parser.add_argument("--max-epochs", default=30, type=int)
parser.add_argument("--train-batch-size", default=64, type=int)
parser.add_argument("--lr-init", default=0.01, type=float)
parser.add_argument("--lr-drop-milestones", nargs="+", default=[8, 16, 24, 27], type=int)
parser.add_argument("--lr-drop-multiplier", default=0.2, type=float)
parser.add_argument("--dataset-dir", default=P("/mnt/d/dataset/MNIST"), type=P)
# parser.add_argument("--dataset-dir", default=P("/data/datasets/MNIST"), type=P)
parser.add_argument("--show-progress-every-n-iters", default=10, type=int)
parser.add_argument("--gradnorm", action="store_true", default=False)

cuda_available = torch.cuda.is_available()
tb_writer = SummaryWriter()
console_logger = get_logger()


def main(args):
    grad_normalizer = SimpleGradNormalizer(lr_init=args.lr_init, alpha=1.5) if args.gradnorm else None
    transforms = {"resize": (32, 32), "normalize": {"mean": [0.131], "std": [0.308]}}
    train_loader, val_loader = prepare_dataloaders(args.dataset_dir, transforms, args.train_batch_size)
    model = prepare_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_init)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, args.lr_drop_milestones, gamma=args.lr_drop_multiplier
    )
    train_progress_visualizer = ProgressVisualizer(
        "train",
        n_iters_per_epoch=math.ceil(len(train_loader.dataset) / args.train_batch_size),
        # show_progress_every_n_iters=args.show_progress_every_n_iters,
        tb_writer=tb_writer,
        console_logger=console_logger,
    )
    val_progress_visualizer = ProgressVisualizer("val", tb_writer=tb_writer, console_logger=console_logger)

    for epoch in range(args.max_epochs):
        train_phase(epoch, model, train_loader, optimizer, train_progress_visualizer, grad_normalizer=grad_normalizer)
        val_phase(epoch, model, val_loader, val_progress_visualizer)
        lr_scheduler.step(epoch + 1)


def prepare_dataloaders(dataset_dir, transforms, train_batch_size):
    """Note:
         batch_size will be only used for train_set. For val_set, batch_size is set to len(val_set)
    """
    train_set = MNIST(
        dataset_dir / "train-images.idx3-ubyte", dataset_dir / "train-labels.idx1-ubyte", transforms=transforms
    )
    val_set = MNIST(
        dataset_dir / "t10k-images.idx3-ubyte", dataset_dir / "t10k-labels.idx1-ubyte", transforms=transforms
    )
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False, num_workers=2)
    return train_loader, val_loader


def prepare_model():
    model = Net(num_classes=10)
    if cuda_available:
        model = model.cuda()
    return model


def train_phase(epoch, model, dataloader, optimizer, visualizer, grad_normalizer=None):
    model.train()
    for i, (images, gt_classes, gt_bboxes, gt_counts) in enumerate(dataloader):
        optimizer.zero_grad()

        if cuda_available:
            images = images.cuda()
            gt_classes = gt_classes.cuda()
            gt_bboxes = gt_bboxes.cuda()
            gt_counts = gt_counts.cuda()

        pred_cls_logits, pred_bboxes, pred_counts = model(images)

        # ------------------ DEBUG ---------------------
        # import torch.nn.functional as F
        # import torch.nn as nn
        # w = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        #
        # # H = model.layer1.conv.weight[0] * images[:, :, :3, :3]
        # # S = H[:, 0, 0, 0]
        # # S = H.view(64, -1)[:, 0]
        # # _loss = ((S - gt_classes.float()) ** 2).mean()
        # # loss = _loss * w
        # # loss.backward(retain_graph=True)
        # # g = torch.autograd.grad(loss, model.layer1.conv.weight, retain_graph=True, create_graph=True)
        #
        # _loss = ((pred_counts - gt_counts) ** 2).mean()
        # loss = _loss * w
        # loss.backward(retain_graph=True)
        # g = torch.autograd.grad(loss, model.counter_head.conv3_conv.weight, retain_graph=True, create_graph=True)
        #
        # # cls_loss = F.cross_entropy(pred_classes, gt_classes, reduction='mean')
        # pred_classes = pred_cls_logits[:, 0].float()
        # # pred_classes.requires_grad = True
        # cls_loss = ((pred_classes - gt_classes.float()) ** 2).mean()
        # loss = cls_loss * w
        # loss.backward(retain_graph=True)
        # g = torch.autograd.grad(loss, model.cls_loc_head.fc.weight, retain_graph=True, create_graph=True)
        # a = 100

        # ------------------ DEBUG ---------------------

        losses = compute_losses(
            pred_cls_logits, pred_bboxes, pred_counts, gt_classes, gt_bboxes, gt_counts, grad_normalizer=grad_normalizer
        )

        performances = measure_performances(pred_cls_logits, pred_bboxes, pred_counts, gt_classes, gt_bboxes, gt_counts)

        if grad_normalizer is not None:
            grad_normalizer.adjust_grad(losses, model)
            loss_weight = {f"w_{i}": w for i, w in enumerate(grad_normalizer.loss_weight)}
            visualizer.display(epoch, i, losses, performances, other_metrics=loss_weight, lr=get_current_lr(optimizer))
        else:
            losses["total_loss"].backward()
            visualizer.display(epoch, i, losses, performances, lr=get_current_lr(optimizer))

        optimizer.step()

    torch.save(model.state_dict(), f"checkpoints/model_epoch{epoch}.pth")


def compute_losses(pred_classes, pred_bboxes, pred_counts, gt_classes, gt_bboxes, gt_counts, grad_normalizer=None):
    losses = {
        "cls_loss": class_loss(pred_classes, gt_classes),
        "bbox_loss": bbox_loss(pred_bboxes, gt_bboxes),
        "counter_loss": counter_loss(pred_counts, gt_counts),
    }

    if grad_normalizer is not None:
        grad_normalizer.adjust_losses(losses)

    losses["total_loss"] = functools.reduce(lambda x, y: x + y, [l for n, l in losses.items()])
    return losses


def measure_performances(pred_classes, pred_bboxes, pred_counts, gt_classes, gt_bboxes, gt_counts):
    performances = {
        "cls_acc": classification_accuracy(pred_classes, gt_classes),
        "cnt_acc": counter_accuracy(pred_counts, gt_counts),
        "bbox_iou": bbox_iou(pred_bboxes, gt_bboxes),
    }
    return performances


def val_phase(epoch, model, dataloader, visualizer):
    model.eval()
    for i, (images, gt_classes, gt_bboxes, gt_counts) in enumerate(dataloader):
        if cuda_available:
            images = images.cuda()
            gt_classes = gt_classes.cuda()
            gt_bboxes = gt_bboxes.cuda()
            gt_counts = gt_counts.cuda()

        pred_classes, pred_bboxes, pred_counts = model(images)
        losses = compute_losses(pred_classes, pred_bboxes, pred_counts, gt_classes, gt_bboxes, gt_counts)
        performances = measure_performances(pred_classes, pred_bboxes, pred_counts, gt_classes, gt_bboxes, gt_counts)

    visualizer.display(epoch, 0, losses, performances)


if __name__ == "__main__":
    main(parser.parse_args())
