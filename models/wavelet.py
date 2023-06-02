import random
import numpy as np
from torch import autograd
from torch import nn
import torch
from util.visualizer import feature_visualization


def get_wav(in_channels, out_channels, pool=True):
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]
    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H
    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)
    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d
    LL = net(in_channels, out_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, out_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, out_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, out_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False
    LL.weight.data = filter_LL.float().unsqueeze(0).expand(out_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(out_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(out_channels, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(out_channels, -1, -1, -1)
    return LL, LH, HL, HH


class WavePool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels, out_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)


def get_wav_two(in_channels, out_channels, pool=True):
    """wavelet decomposition using conv2d"""
    """小波分解"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    #  print(filter_LL.size())
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d
    LL = net(in_channels, out_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=out_channels)
    LH = net(in_channels, out_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=out_channels)
    HL = net(in_channels, out_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=out_channels)
    HH = net(in_channels, out_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=out_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return LL, LH, HL, HH


class WaveUnpool(nn.Module):
    def __init__(self, in_channels, out_channels, option_unpool='cat5', visualize_stage=0):
        super(WaveUnpool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.option_unpool = option_unpool
        self.LL, self.LH, self.HL, self.HH = get_wav_two(self.in_channels, self.out_channels, pool=False)
        self.stage = visualize_stage

    def forward(self, LL, LH, HL, HH, original=None):
        if self.option_unpool == 'sum':
            return self.LL(LL) + self.LH(LH) + self.HL(HL) + self.HH(HH)
        elif self.option_unpool == 'cat5' and original is not None:
            return torch.cat([self.LL(LL), self.LH(LH), self.HL(HL), self.HH(HH), original], dim=1)
        elif self.option_unpool == 'add_high':
            return self.LH(LH) + self.HL(HL) + self.HH(HH)
        elif self.option_unpool == 'add_low':
            return self.LL(LL)
        else:
            raise NotImplementedError


if __name__ == '__main__':
    get_wav(in_channels=3,out_channels=6,pool=True)
