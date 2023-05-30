import torch
import torch.nn as nn
from models.dptn_networks import modules
from models.dptn_networks.base_network import BaseNetwork
from models.spade_networks.architecture import SPADEResnetBlock, SPAINResnetBlock
from models.spade_networks.normalization import get_nonspade_norm_layer

import torch.nn.functional as F

class Resblock(nn.Module) :
    def __init__(self, opt, input_nc, output_nc):
        super(Resblock, self).__init__()
        self.opt = opt

        kw = 3
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)

        self.conv1 = norm_layer(nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1))
        self.conv2 = norm_layer(nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1))
        self.shortcut = norm_layer(nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1))
    def forward(self, x):
        out = self.conv2(self.conv1(x)) + self.shortcut(x)
        return out

class ImageDecoder(nn.Module) :
    def __init__(self, opt):
        super(ImageDecoder, self).__init__()
        self.opt = opt
        nf = opt.ngf

        self.layer1 = Resblock(opt, 3, 16 * nf)
        self.layer2 = Resblock(opt, 16 * nf, 8 * nf)
        self.layer3 = Resblock(opt, 8 * nf, 4 * nf)
        self.layer4 = Resblock(opt, 4 * nf, 2 * nf)

        final_nc = 2 * nf

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.up(x) # 32x32 -> 64x64
        x = self.layer3(x)
        x = self.up(x) # 64x64 -> 128x128
        x = self.layer4(x)
        x = self.up(x) # 128x128 -> 256x256

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x
