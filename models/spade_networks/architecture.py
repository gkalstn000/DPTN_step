"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm
from models.spade_networks.normalization import SPADE, SPADEAttn


# ResNet block that uses SPADE.
# It differs from the ResNet block of pix2pixHD in that
# it takes in the segmentation map as input, learns the skip connection if necessary,
# and applies normalization first and then convolution.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was inspired from https://github.com/LMescheder/GAN_stability.
class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt):
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

        # define normalization layers
        if opt.type_En_c.lower() == 'spade' :
            cond_norm = SPADE
        elif opt.type_En_c.lower() == 'spadeattn' :
            cond_norm = SPADEAttn
        else :
            raise Exception(f'{opt.type_En_c} is unrecognized cond norm type')

        input_nc = opt.image_nc + 2 * opt.pose_nc # image channel + map channer * 2
        self.norm_0 = cond_norm(opt.norm, fin, opt.pose_nc)
        self.norm_1 = cond_norm(opt.norm, fmiddle, opt.pose_nc)
        if self.learned_shortcut:
            self.norm_s = cond_norm(opt.norm, fin, opt.pose_nc)

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

if __name__ == "__main__" :
    conv0 = nn.Conv2d(10, 10, kernel_size=3, padding=1)