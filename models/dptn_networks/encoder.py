import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dptn_networks import modules
from models.dptn_networks.base_network import BaseNetwork
from models.spade_networks.architecture import SPADEResnetBlock
from models.spade_networks.normalization import get_nonspade_norm_layer
import math
from models.dptn_networks import encoder
import numpy as np


class NoiseEncoder(BaseNetwork):
    def __init__(self, opt):
        super(NoiseEncoder, self).__init__()
        self.opt = opt

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf
        input_nc = opt.image_nc
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)

        self.layer1 = norm_layer(nn.Conv2d(input_nc, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 2, kw, stride=1, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 2, ndf * 1, kw, stride=1, padding=pw))
        self.layer6 = norm_layer(nn.Conv2d(ndf * 1, ndf // 2, kw, stride=1, padding=pw))
        self.layer7 = norm_layer(nn.Conv2d(ndf // 2, ndf // 4, kw, stride=1, padding=pw))
        self.layer8 = norm_layer(nn.Conv2d(ndf // 4, 3, kw, stride=1, padding=pw))

        self.actvn = nn.LeakyReLU(0.2, False)

        self.so = s0 = 4
        self.fc_mu = FC_layer(opt, 3)
        self.fc_var = FC_layer(opt, 3)
    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.layer1(x)              # 256x256 -> 128x128
        x = self.layer2(self.actvn(x))  # 128x128 -> 64x64
        x = self.layer3(self.actvn(x))  # 64x64 -> 32x32
        x = self.layer4(self.actvn(x))  # 64x64 -> 32x32
        x = self.layer5(self.actvn(x))  # 64x64 -> 32x32
        x = self.layer6(self.actvn(x))  # 64x64 -> 32x32
        x = self.layer7(self.actvn(x))  # 64x64 -> 32x32
        x = self.layer8(self.actvn(x))  # 64x64 -> 32x32
        x = self.actvn(x)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar

class FC_layer(nn.Module) :
    def __init__(self, opt, init_dim):
        super(FC_layer, self).__init__()
        self.opt = opt
        self.so = s0 = 32

        self.layer1 = nn.Sequential(
            nn.Linear(init_dim * s0 * s0, self.opt.z_dim),
            # nn.BatchNorm1d(self.opt.z_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5))

        self.layer2 = nn.Sequential(
            nn.Linear(self.opt.z_dim, self.opt.z_dim),
            # nn.BatchNorm1d(self.opt.z_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5))

        self.layer3 = nn.Sequential(
            nn.Linear(self.opt.z_dim, self.opt.z_dim),
            # nn.BatchNorm1d(self.opt.z_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5))

        self.layer4 = nn.Sequential(
            nn.Linear(self.opt.z_dim, self.opt.z_dim),
            # nn.BatchNorm1d(self.opt.z_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5))

        self.layer5 = nn.Sequential(
            nn.Linear(self.opt.z_dim, self.opt.z_dim),
            # nn.BatchNorm1d(self.opt.z_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5))

        self.layer6 = nn.Sequential(
            nn.Linear(self.opt.z_dim, self.opt.z_dim),
            # nn.BatchNorm1d(self.opt.z_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5))

        self.layer7 = nn.Sequential(
            nn.Linear(self.opt.z_dim, self.opt.z_dim),
            # nn.BatchNorm1d(self.opt.z_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5))

        self.layer8 = nn.Sequential(
            nn.Linear(self.opt.z_dim, self.opt.z_dim),
            # nn.BatchNorm1d(self.opt.z_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.opt.z_dim, self.opt.z_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)

        return x