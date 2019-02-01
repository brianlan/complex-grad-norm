import argparse
from pathlib import Path as P
import functools

import torch
from torch.utils.data import DataLoader

from src.model import Net
from src.dataset import MNIST
from src.loss import class_loss, bbox_loss, counter_loss


parser = argparse.ArgumentParser("Training script")
parser.add_argument("--max-epochs", default=10, type=int)
parser.add_argument("--batch-size", default=64, type=int)
parser.add_argument("--lr-init", default=0.001, type=float)
parser.add_argument("--lr-drop-milestones", nargs="+", default=[4, 8], type=int)
parser.add_argument("--lr-drop-multiplier", default=0.1, type=float)
parser.add_argument("--dataset-dir", default=P("/mnt/d/dataset/MNIST"), type=P)

cuda_available = torch.cuda.is_available()


def main(args):
    ####################
    # Define Transforms
    ####################
    transforms = {"resize": (32, 32), "normalize": {"mean": [0.131], "std": [0.308]}}

    ####################
    # Define Dataset
    ####################
    train_set = MNIST(
        args.dataset_dir / "train-images.idx3-ubyte",
        args.dataset_dir / "train-labels.idx1-ubyte",
        transforms=transforms,
    )
    val_set = MNIST(
        args.dataset_dir / "t10k-images.idx3-ubyte", args.dataset_dir / "t10k-labels.idx1-ubyte", transforms=transforms
    )

    ####################
    # Define DataLoader
    ####################
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    ####################
    # Define Network
    ####################
    net = Net(num_classes=10)
    if cuda_available:
        net = net.cuda()

    ####################
    # Define Optimizer
    ####################
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr_init)

    ##################################
    # Define Learning Rate Scheduler
    ##################################
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, args.lr_drop_milestones, gamma=args.lr_drop_multiplier
    )

    for epoch in range(args.max_epochs):
        train_phase(epoch, net, train_loader, optimizer)
        val_phase(epoch, net, val_loader)
        lr_scheduler.step(epoch + 1)
        # performance = calculate_performance(pred, labels)


def train_phase(epoch, net, dataloader, optimizer):
    net.train()
    for i, (images, gt_classes, gt_bboxes, gt_counts) in enumerate(dataloader):
        optimizer.zero_grad()

        if cuda_available:
            images = images.cuda()
            gt_classes = gt_classes.cuda()
            gt_bboxes = gt_bboxes.cuda()
            gt_counts = gt_counts.cuda()

        pred_classes, pred_bboxes, pred_counts = net(images)
        losses = compute_losses(pred_classes, pred_bboxes, pred_counts, gt_classes, gt_bboxes, gt_counts)
        losses["total_loss"].backward()

        print_progress(epoch, i, losses)

        optimizer.step()


def print_progress(epoch, it, losses, print_every=50):
    if (it + 1) % print_every == 0:
        print("-" * 60)
        print(f"Epoch {epoch}, Iteration {iter + 1}")
        print("-" * 30)
        print("\n".join([f"{n:16s}: {l.item():.4f}" for n, l in losses.items()]))


def compute_losses(pred_classes, pred_bboxes, pred_counts, gt_classes, gt_bboxes, gt_counts):
    losses = {
        "cls_loss": class_loss(pred_classes, gt_classes),
        "bbox_loss": bbox_loss(pred_bboxes, gt_bboxes),
        "counter_loss": counter_loss(pred_counts, gt_counts),
    }
    losses["total_loss"] = functools.reduce(lambda x, y: x + y, [l for n, l in losses.items()])
    return losses


def val_phase(epoch, net, dataloader):
    net.eval()
    for i, (images, classes, bboxes, counts) in enumerate(dataloader):
        if cuda_available:
            images = images.cuda()
            classes = classes.cuda()
            bboxes = bboxes.cuda()
            counts = counts.cuda()

        pred = net(images)


if __name__ == "__main__":
    main(parser.parse_args())
