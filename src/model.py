import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBatchnormReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


def flatten(x):
    return x.view(x.shape[0], -1)


class Concat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *input, dim=1):
        return torch.cat(input, dim=dim)


class ClassificationLocationHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc = nn.Linear(96 * 4 * 4, 1008)
        self.fc_bn = nn.BatchNorm1d(1008)
        self.fc_relu = nn.ReLU()
        self.classifier = nn.Linear(1008, num_classes)
        self.regressor = nn.Linear(1008, 4)

    def forward(self, x):
        x = self.fc(x)
        x = self.fc_bn(x)
        x = self.fc_relu(x)
        return self.classifier(x), self.regressor(x)


class CounterHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBatchnormReLU(48, 64, 3, padding=1)
        self.conv2 = ConvBatchnormReLU(64, 64, 3, padding=1)
        self.conv3 = ConvBatchnormReLU(64, 64, 3, stride=2, padding=1)
        self.conv3_conv = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv3_relu = nn.ReLU()

        self.fc = nn.Linear(64 * 8 * 8, 1008)
        self.fc_bn = nn.BatchNorm1d(1008)
        self.fc_relu = nn.ReLU()
        self.regressor = nn.Linear(1008, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # self.before_conv3 = x
        x = self.conv3(x)
        # x = self.conv3_conv(x)
        # x = self.conv3_bn(x)
        # x = self.conv3_relu(x)
        x = flatten(x)
        # self.before_flatten = x
        # x = x.view(64, 4096)
        # self.after_flatten = x
        x = self.fc_relu(self.fc_bn(self.fc(x)))
        x = self.regressor(x)
        return x


class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layer1 = ConvBatchnormReLU(1, 16, 3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2_1 = ConvBatchnormReLU(16, 32, 3, padding=1)
        self.layer2_2 = ConvBatchnormReLU(32, 32, 3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer3_1 = ConvBatchnormReLU(32, 48, 3, padding=1)
        self.layer3_2 = ConvBatchnormReLU(48, 48, 3, padding=1)
        self.layer3_3 = ConvBatchnormReLU(48, 48, 3, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.concat1 = Concat()
        self.concat2 = Concat()

        self.cls_loc_head = ClassificationLocationHead(num_classes)
        self.counter_head = CounterHead()

    def forward(self, x):
        x = self.layer1(x)
        fmap1 = x = self.maxpool1(x)
        x = self.layer2_1(x)
        x = self.layer2_2(x)
        fmap2 = x = self.maxpool2(x)
        x = self.layer3_1(x)
        x = self.layer3_2(x)
        x = self.layer3_3(x)
        fmap3 = self.maxpool3(x)

        concat1 = self.concat1(
            F.avg_pool2d(fmap1, kernel_size=4, stride=4), F.avg_pool2d(fmap2, kernel_size=2, stride=2), fmap3, dim=1
        )

        concat2 = self.concat2(fmap1, F.upsample(fmap2, scale_factor=2), dim=1)

        cls, bbox = self.cls_loc_head(flatten(concat1))
        cnt = self.counter_head(concat2)
        return cls, bbox, cnt
