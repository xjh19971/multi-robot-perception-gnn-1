import torch
import torch.nn as nn
import math

class TransBlock(nn.Module):

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0, kernel_size=3):
        super().__init__()
        if stride == 2:
            self.transconv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))
            self.Upsample = nn.Upsample(scale_factor=(stride, stride), mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.transconv1(x)
        x = self.Upsample(x)
        return x