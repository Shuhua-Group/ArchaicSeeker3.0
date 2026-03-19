import torch
import torch.nn as nn
import torchmetrics
from torch.nn import functional as F
from .mudules import ResidualConv, Upsample, Down, Up, DoubleConv, OutConv

import numpy
import cv2


class ResUnet(nn.Module):
    def __init__(self, channel, n_classes, filters=[16, 32, 64, 128]):
        super(ResUnet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv1d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm1d(filters[0]),
            nn.ReLU(),
            nn.Conv1d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv1d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv1d(filters[0], n_classes, 1, 1),
        )


    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x) # 65666
        
        x1_p, l1, r1 = pad_tensor(x1, 2) # 65666

        x2 = self.residual_conv_1(x1_p) # 32833
        
        x2_p, l2, r2 = pad_tensor(x2, 2) # 32834
        
        x3 = self.residual_conv_2(x2_p) # 16417

        x3_p, l3, r3 = pad_tensor(x3, 2) # 16418
        # Bridge
        x4 = self.bridge(x3_p) # 8209
        # Decode

        x4 = self.upsample_1(x4) # 16418

        x4_pb = pad_tensor_back(x4, l2, r3) # 16417

        x5 = torch.cat([x4_pb, x3], dim=1) # 16417

        x6 = self.up_residual_conv1(x5) # 16417
        x6 = self.upsample_2(x6) # 32834

        x6_pb = pad_tensor_back(x6, l2, r2) # 32833

        x7 = torch.cat([x6_pb, x2], dim=1) # 32833

        x8 = self.up_residual_conv2(x7) # 32833
        x8 = self.upsample_3(x8) # 65666

        x8_pb = pad_tensor_back(x8, l1, r1) # 65666

        x9 = torch.cat([x8_pb, x1], dim=1) # 65666

        x10 = self.up_residual_conv3(x9) # 65666

        output = self.output_layer(x10) # 65666

        return output



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


def pad_tensor(_in, divide):
    
    length = _in.shape[2]

    res = length % divide
    
    if res != 0:
        div = divide - res
        pad_left = int(div / 2)
        pad_right = int(div - pad_left)
        padding = nn.ReflectionPad1d((pad_left, pad_right))
        _in = padding(_in).data
    else:
        pad_left = 0
        pad_right = 0

    return _in, pad_left, pad_right

def pad_tensor_back(_in, pad_left, pad_right):
    length = _in.shape[2]
    return _in[:,:, pad_left: length - pad_right]