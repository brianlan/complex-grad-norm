import torch
import numpy as np
import scipy.ndimage.measurements as measurements
from PIL import Image
import torchvision.transforms.functional as F


def get_num_regions(img):
    return torch.tensor(measurements.label(img)[1], dtype=torch.float)


def get_digit_bbox(img):
    if isinstance(img, Image.Image):
        img = np.array(img)
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    _, y_ind, x_ind = np.where(img > 0)
    return torch.tensor([x_ind.min(), y_ind.min(), x_ind.max(), y_ind.max()], dtype=torch.float)


class MNIST:
    @staticmethod
    def _skip_image_header(file):
        """ header contains:
        1. 32 bit integer magic number
        2. 32 bit integer number of images
        3. 32 bit number of rows
        4. 32 bit number of columns
        """
        [file.read(4) for i in range(4)]

    @staticmethod
    def _skip_label_header(file):
        """ header contains:
        1. 32 bit integer magic number
        2. 32 bit integer number of items
        """
        [file.read(4) for i in range(2)]

    def read_images(self, path):
        with open(path, "rb") as f:
            MNIST._skip_image_header(f)
            _images = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)
        for img in _images:
            self.images.append(Image.fromarray(img))

    def read_labels(self, path):
        with open(path, "rb") as f:
            MNIST._skip_label_header(f)
            _labels = np.frombuffer(f.read(), np.uint8)
        self.labels = torch.tensor(_labels, dtype=torch.long)

    def __init__(self, image_path, label_path, transforms=None):
        self.images = []
        self.labels = None
        self.read_images(image_path)
        self.read_labels(label_path)
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, cls = self.images[idx], self.labels[idx]
        if self.transforms.get('resize') is not None:
            img = F.resize(img, self.transforms['resize'])
        img = F.to_tensor(img)

        bbox = get_digit_bbox(img)
        cnt = get_num_regions(img)

        if self.transforms.get('normalize') is not None:
            img = F.normalize(img, self.transforms['normalize']['mean'], self.transforms['normalize']['std'])

        return img, cls, bbox, cnt
