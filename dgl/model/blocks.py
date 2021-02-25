import torch
import torch.nn as nn
import math

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            nn.Dropout(p=dropout, inplace=True)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU()(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
            nn.Dropout(p=dropout, inplace=True)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU()(self.residual_function(x) + self.shortcut(x))

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
        # # residual function
        # if stride == 2:
        #     self.transconv1 = nn.Sequential(
        #         nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1,
        #                            output_padding=1, bias=False),
        #         nn.BatchNorm2d(out_channels),
        #         nn.ReLU(inplace=True))
        # else:
        #     self.transconv1 = nn.Sequential(
        #         nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1,
        #                            bias=False),
        #         nn.BatchNorm2d(out_channels),
        #         nn.ReLU(inplace=True))
        # self.transconv2 = nn.Sequential(
        #     nn.ConvTranspose2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(out_channels * BasicBlock.expansion),
        #     nn.Dropout(p=dropout, inplace=True))
        #
        # # shortcut
        # self.shortcut = nn.Sequential()
        #
        # # the shortcut output dimension is not the same with residual function
        # # use 1*1 convolution to match the dimension
        # if stride != 1:
        #     self.shortcut = nn.Sequential(
        #         nn.ConvTranspose2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride,
        #                            bias=False, output_padding=1),
        #         nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        #     )
        # elif in_channels != BasicBlock.expansion * out_channels:
        #     self.shortcut = nn.Sequential(
        #         nn.ConvTranspose2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride,
        #                            bias=False),
        #         nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        #     )

    def forward(self, x):
        x = self.transconv1(x)
        x = self.Upsample(x)
        return x
        # x_d = self.transconv2(self.transconv1(x))
        # x_p = self.shortcut(x)
        # return nn.ReLU()(x_d + x_p)