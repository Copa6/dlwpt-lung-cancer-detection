import math
import torch
from torch import nn
import torch.nn.functional as F


class LunaModel(nn.Module):

    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()
        self.bn = nn.BatchNorm3d(1)

        self.block1 = LunaBlock(in_channels, conv_channels)
        self.block2 = LunaBlock(conv_channels, conv_channels*2)
        self.block3 = LunaBlock(conv_channels*2, conv_channels*4)
        self.block4 = LunaBlock(conv_channels*4, conv_channels*8)

        self.fc = nn.Linear(64*3*3*2, 2)
        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv3d}:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        norm_batch = self.bn(input_batch)
        conv_out = self.block1(norm_batch)
        conv_out = self.block2(conv_out)
        conv_out = self.block3(conv_out)
        conv_out = self.block4(conv_out)

        flat_out = conv_out.view(conv_out.size(0), -1)
        lin_out = self.fc(flat_out)
        softmax_out = self.softmax(lin_out)

        return lin_out, softmax_out


class LunaBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels=conv_channels, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv3d(
            conv_channels, out_channels=conv_channels, kernel_size=3, padding=1, bias=True)
        self.maxpool = nn.MaxPool3d(2, 2)

    def forward(self, in_batch):
        conv_out = F.relu(self.conv1(in_batch))
        conv_out = F.relu(self.conv2(conv_out))
        block_out = self.maxpool(conv_out)

        return block_out

