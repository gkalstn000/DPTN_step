"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm
from models.spade_networks.normalization import SPADE, AdaIN


# ResNet block that uses SPADE.
# It differs from the ResNet block of pix2pixHD in that
# it takes in the segmentation map as input, learns the skip connection if necessary,
# and applies normalization first and then convolution.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was inspired from https://github.com/LMescheder/GAN_stability.
class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt, norm_nc):
        '''
        :param fin: input dim of main feature map
        :param fout: output dim of main feature map
        :param opt: options
        :param norm_nc: norm input dim
        '''
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        self.conv_0 = spectral_norm(self.conv_0)
        self.conv_1 = spectral_norm(self.conv_1)
        if self.learned_shortcut:
            self.conv_s = spectral_norm(self.conv_s)

        cond_norm = SPADE
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = cond_norm(spade_config_str, fin, norm_nc)
        self.norm_1 = cond_norm(spade_config_str, fmiddle, norm_nc)
        if self.learned_shortcut:
            self.norm_s = cond_norm(spade_config_str, fin, norm_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, texture_information):
        x_s = self.shortcut(x, texture_information)

        dx = self.conv_0(self.actvn(self.norm_0(x, texture_information)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, texture_information)))

        out = x_s + dx

        return out

    def shortcut(self, x, texture_information):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, texture_information))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class SPAINResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt, norm_nc):
        '''
        :param fin: input dim of main feature map
        :param fout: output dim of main feature map
        :param opt: options
        :param norm_nc: norm input dim
        '''
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        self.conv_0 = spectral_norm(self.conv_0)
        self.conv_1 = spectral_norm(self.conv_1)
        if self.learned_shortcut:
            self.conv_s = spectral_norm(self.conv_s)

        config_str = opt.norm_G.replace('spectral', '')
        self.spade_norm_0 = SPADE(config_str, fin, norm_nc)
        self.spade_norm_1 = SPADE(config_str, fmiddle, norm_nc)
        if self.learned_shortcut:
            self.spade_norm_s = SPADE(config_str, fin, norm_nc)

        self.adain_norm_0 = AdaIN(config_str, fin, opt.z_dim)
        self.adain_norm_1 = AdaIN(config_str, fmiddle, opt.z_dim)
        if self.learned_shortcut :
            self.adain_norm_s = AdaIN(config_str, fin, opt.z_dim)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, pose_information, texture_information):
        x_s = self.shortcut(x, pose_information, texture_information)

        dx = self.actvn(self.spade_norm_0(x, pose_information))
        dx = self.conv_0(self.actvn(self.adain_norm_0(dx, texture_information)))
        dx = dx + torch.randn_like(dx, device=dx.device)

        dx = self.actvn(self.spade_norm_1(dx, pose_information))
        dx = self.conv_1(self.actvn(self.adain_norm_1(dx, texture_information)))
        dx = dx + torch.randn_like(dx, device=dx.device)

        out = x_s + dx

        return out

    def shortcut(self, x, pose_information, texture_information):
        if self.learned_shortcut:
            x_s = self.spade_norm_s(x, pose_information)
            x_s = self.conv_s(self.adain_norm_s(x_s, texture_information))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


if __name__ == "__main__" :
    conv0 = nn.Conv2d(10, 10, kernel_size=3, padding=1)