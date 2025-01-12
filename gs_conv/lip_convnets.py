# code modified from https://github.com/singlasahil14/SOC
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from skew_ortho_conv import SOC, GS_SOC, PermutedSOC, GS_SOC_Accelerated, LPRSOC

from custom_activations import *
from utils import conv_mapping, activation_mapping


class LipBlock(nn.Module):
    def __init__(self, in_planes, planes, conv_layer, activation, stride=1, kernel_size=3, groups=1, paired=False):
        super(LipBlock, self).__init__()
        self.conv = conv_layer(in_planes, planes*stride, kernel_size=kernel_size, 
                               stride=stride, padding=kernel_size//2, groups=groups, paired=paired)
        self.activation = activation_mapping(activation)

    def forward(self, x):
        x = self.activation(self.conv(x))
        return x
        
class LipConvNet(nn.Module):
    def __init__(self, conv_name, activation, init_channels=32, block_size=1, 
                 num_classes=10, input_side=32, groups=1, paired=False):
        super(LipConvNet, self).__init__()
        self.in_planes = 3
        
        conv_layer = conv_mapping[conv_name]
        assert type(block_size) == int

        self.layer1 = self._make_layer(init_channels, block_size, conv_layer, activation,
                                        stride=2, kernel_size=3, groups=groups, paired=paired)
        self.layer2 = self._make_layer(self.in_planes, block_size, conv_layer, activation,
                                        stride=2, kernel_size=3, groups=groups, paired=paired)
        self.layer3 = self._make_layer(self.in_planes, block_size, conv_layer, activation,
                                        stride=2, kernel_size=3, groups=groups, paired=paired)
        self.layer4 = self._make_layer(self.in_planes, block_size, conv_layer, activation,
                                        stride=2, kernel_size=3, groups=groups, paired=paired)
        self.layer5 = self._make_layer(self.in_planes, block_size, conv_layer, activation,
                                        stride=2, kernel_size=1, groups=groups, paired=paired)
        
        flat_size = input_side // 32
        flat_features = flat_size * flat_size * self.in_planes
        self.last_layer = SOC(flat_features, num_classes, 
                                        kernel_size=1, stride=1, groups=1)

    def _make_layer(self, planes, num_blocks, conv_layer, activation,
                    stride, kernel_size, groups, paired):
        strides = [1]*(num_blocks-1) + [stride]
        kernel_sizes = [3]*(num_blocks-1) + [kernel_size]
        layers = []
        for stride, kernel_size in zip(strides, kernel_sizes):
            layers.append(LipBlock(self.in_planes, planes, conv_layer, activation,
                                   stride, kernel_size, groups, paired))
            self.in_planes = planes * stride
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.last_layer(x)
        x = x.view(x.shape[0], -1)
        return x