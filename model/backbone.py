
import torch
import torch.nn as nn
from model.blocks import BasicBlock, BottleNeck


class ResNet(nn.Module):

    def __init__(self, block_type_index, block_num_index, opt):
        super().__init__()
        self.opt = opt
        self.block_type = [BasicBlock, BottleNeck]
        self.block_num = [[2, 2, 2, 2],
                          [3, 4, 6, 3],
                          [3, 4, 6, 3],
                          [3, 4, 23, 3],
                          [3, 8, 36, 3]]
        self._construct_net(self.block_type[block_type_index], self.block_num[block_num_index])

    def _construct_net(self, block, num_block, num_classes=100):
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=1, bias=False, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 2)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, self.opt.dropout))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output_list=[]
        output = self.conv1(x)
        output = self.conv2_x(output)
        output_list.append(output)
        output = self.conv3_x(output)
        output_list.append(output)
        output = self.conv4_x(output)
        output_list.append(output)
        output = self.conv5_x(output)
        output_list.append(output)
        return output_list

def resnet_wrapper(name, opt):
    if name == "resnet18":
        return ResNet(0, 0, opt)
    elif name == "resnet34":
        return ResNet(0, 1, opt)
    elif name == "resnet50":
        return ResNet(1, 2, opt)
    elif name == "resnet101":
        return ResNet(1, 3, opt)
    elif name == "resnet152":
        return ResNet(1, 4, opt)
