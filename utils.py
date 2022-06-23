import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable
from typing import Optional
from torch.utils.data import Dataset, TensorDataset
from torch import Tensor
from numpy import ndarray


class TensorDatasetForKfold(Dataset):

    def __init__(self, *tensors: Tensor, valid_fold, status) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors
        self.valid_fold = valid_fold
        self.N = tensors[0].size(0)
        self.statues = status

    def __getitem__(self, index):
        if self.statues == 'train':
            return tuple(
                torch.from_numpy(
                    torch.cat(
                        (tensor[0: int(self.N * self.valid_fold / 5)],
                         tensor[int(self.N * (self.valid_fold + 1) / 5): self.N])))[index]
                for tensor in self.tensors)
        if self.statues == 'valid':
            return tuple(
                tensor[int(self.N * self.valid_fold / 5): int(self.N * (self.valid_fold + 1) / 5)][index]
                for tensor in self.tensors)
        if self.statues == 'test':
            return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        if self.statues == 'train':
            return self.N - (int(self.N * (self.valid_fold + 1) / 5) - int(self.N * self.valid_fold / 5))
        if self.statues == 'valid':
            return int(self.N * (self.valid_fold + 1) / 5) - int(self.N * self.valid_fold / 5)
        if self.statues == 'test':
            return self.N


class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))
        # super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, True, self.momentum,
                            self.eps)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = ContBatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=dilation * multi_grid,
                               dilation=dilation * multi_grid, bias=False)
        self.bn2 = ContBatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = ContBatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out


def torch_dice_coef_loss(y_pred, y_true, smooth=1.):
    return 1. - dice_coef(y_pred, y_true)


def iou(y_pred, y_true):
    overlap = torch.sum((y_pred > 0.5) * (y_true > 0.5), axis=[1, 2, 3, 4])
    union = torch.sum((y_pred > 0.5) + (y_true > 0.5), axis=[1, 2, 3, 4])
    IoU = torch.mean(overlap / union, axis=0)
    return torch.mean(IoU)


def dice_coef(y_pred, y_true, smooth=1.):
    intersection = torch.sum(y_true * y_pred, axis=[1, 2, 3, 4])
    union = torch.sum(y_true, axis=[1, 2, 3, 4]) + torch.sum(y_pred, axis=[1, 2, 3, 4])
    dice = torch.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice
