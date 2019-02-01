import argparse
from pathlib import Path as P
import functools

import torch
from torch.utils.data import DataLoader

from src.model import Net
from src.dataset import MNIST
from src.loss import class_loss, bbox_loss, counter_loss
from src.measure import classification_accuracy, counter_accuracy, bbox_iou

parser = argparse.ArgumentParser("Training script")
parser.add_argument("--max-epochs", default=10, type=int)
parser.add_argument("--batch-size", default=64, type=int)
parser.add_argument("--lr-init", default=0.001, type=float)
parser.add_argument("--lr-drop-milestones", nargs="+", default=[4, 8], type=int)
parser.add_argument("--lr-drop-multiplier", default=0.1, type=float)
# parser.add_argument("--dataset-dir", default=P("/mnt/d/dataset/MNIST"), type=P)
parser.add_argument("--dataset-dir", default=P("/data/datasets/MNIST"), type=P)

cuda_available = torch.cuda.is_available()


def main(args):
    transforms = {"resize": (32, 32), "normalize": {"mean": [0.131], "std": [0.308]}}
    train_loader, val_loader = prepare_dataloaders(args.dataset_dir, transforms, args.batch_size)
    model = prepare_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_init)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, args.lr_drop_milestones, gamma=args.lr_drop_multiplier
    )

    for epoch in range(args.max_epochs):
        train_phase(epoch, model, train_loader, optimizer)
        val_phase(epoch, model, val_loader)
        lr_scheduler.step(epoch + 1)


def prepare_dataloaders(dataset_dir, transforms, batch_size):
    train_set = MNIST(
        dataset_dir / "train-images.idx3-ubyte", dataset_dir / "train-labels.idx1-ubyte", transforms=transforms
    )
    val_set = MNIST(
        dataset_dir / "t10k-images.idx3-ubyte", dataset_dir / "t10k-labels.idx1-ubyte", transforms=transforms
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader


def prepare_model():
    model = Net(num_classes=10)
    if cuda_available:
        model = model.cuda()
    return model


def train_phase(epoch, model, dataloader, optimizer):
    model.train()
    for i, (images, gt_classes, gt_bboxes, gt_counts) in enumerate(dataloader):
        optimizer.zero_grad()

        if cuda_available:
            images = images.cuda()
            gt_classes = gt_classes.cuda()
            gt_bboxes = gt_bboxes.cuda()
            gt_counts = gt_counts.cuda()

        pred_classes, pred_bboxes, pred_counts = model(images)
        losses = compute_losses(pred_classes, pred_bboxes, pred_counts, gt_classes, gt_bboxes, gt_counts)
        losses["total_loss"].backward()

        performances = measure_performances(pred_classes, pred_bboxes, pred_counts, gt_classes, gt_bboxes, gt_counts)
        print_progress(epoch, i, losses, performances)

        optimizer.step()

    torch.save(model.state_dict(), f"checkpoints/model_epoch{epoch}.pth")


def print_progress(epoch, it, losses, performances, print_every=50):
    if (it + 1) % print_every == 0:
        print("-" * 60)
        print(f"Epoch {epoch}, Iteration {it + 1}")
        print("." * 30)
        print("\n".join([f"{n:16s}: {l.item():.4f}" for n, l in losses.items()]))
        print("." * 30)
        print("\n".join([f"{n:16s}: {p.item():.4f}" for n, p in performances.items()]))


def compute_losses(pred_classes, pred_bboxes, pred_counts, gt_classes, gt_bboxes, gt_counts):
    losses = {
        "cls_loss": class_loss(pred_classes, gt_classes),
        "bbox_loss": bbox_loss(pred_bboxes, gt_bboxes),
        "counter_loss": counter_loss(pred_counts, gt_counts),
    }
    losses["total_loss"] = functools.reduce(lambda x, y: x + y, [l for n, l in losses.items()])
    return losses


def measure_performances(pred_classes, pred_bboxes, pred_counts, gt_classes, gt_bboxes, gt_counts):
    performances = {
        "cls_acc": classification_accuracy(pred_classes, gt_classes),
        "cnt_acc": counter_accuracy(pred_counts, gt_counts),
        "bbox_iou": bbox_iou(pred_bboxes, gt_bboxes),
    }
    return performances


def val_phase(epoch, model, dataloader):
    model.eval()
    for i, (images, classes, bboxes, counts) in enumerate(dataloader):
        if cuda_available:
            images = images.cuda()
            classes = classes.cuda()
            bboxes = bboxes.cuda()
            counts = counts.cuda()

        pred = model(images)


if __name__ == "__main__":
    main(parser.parse_args())
