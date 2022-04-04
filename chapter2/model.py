import torch.nn as nn 
import numpy as np 
import torch
from torch.nn.functional import F


def accuracy(targets, predictions):
    return np.sum(targets == predictions) / float(len(targets))

def outputShapeConv2d(h, w, filter, pooling=2):
    kernel_size = filter.kernel_size
    stride = filter.stride
    padding = filter.padding
    dilation = filter.dilation

    h = np.floor((h + 2 * padding[0] - dilation[0] * (kernel_size[0] -1) - 1)/stride[0] + 1)
    w = np.floor((w + 2 * padding[1] - dilation[1] * (kernel_size[1] -1) - 1)/stride[1] + 1)

    if pooling:
        h /= pooling
        w /= pooling
    return int(h), int(w)


class Net(nn.Module):
    def __init__(self):
        self.image_shape = (96, 96)
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, 
            out_channels=8,
            kernel_size=3
        )
        h, w = outputShapeConv2d(self.image_shape[1], self.image_shape[0], self.conv1)

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3
        )
        h, w = outputShapeConv2d(h, w, self.conv2) 

        self.conv3 = nn.Conv2d(
            in_channels=16, 
            out_channels=32,
            kernel_size=3
        )
        h, w = outputShapeConv2d(h, w, self.conv3)

        self.conv4 = nn.Conv2d(
            in_channels=32, 
            out_channels=64, 
            kernel_size=3
        )
        h, w = outputShapeConv2d(h, w, self.conv4)

        self.flatten = h * w * 64
        self.fc1 = nn.Linear(self.flatten, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, self.flatten)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, 0.25, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
        