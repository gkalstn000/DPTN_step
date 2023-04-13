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

class SpadeEncoder(BaseNetwork) :
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.layers = opt.layers_g
        nf = opt.ngf
        nonlinearity = modules.get_nonlinearity_layer(activation_type=opt.activation)
        norm_layer = modules.get_norm_layer(norm_type=opt.norm)

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        input_nc = opt.image_nc
        norm_nc = opt.pose_nc
        self.head_0 = SPADEResnetBlock(input_nc, nf, opt, norm_nc)

        self.mult = 1
        for i in range(self.layers - 1) :
            mult_prev = self.mult
            self.mult = min(2 ** (i + 1), opt.img_f // opt.ngf)
            block = SPADEResnetBlock(opt.ngf * mult_prev, opt.ngf * self.mult, opt, norm_nc)
            setattr(self, 'down' + str(i), block)

        # ResBlocks
        for i in range(opt.num_blocks):
            block = modules.ResBlock(opt.ngf * self.mult, opt.ngf * self.mult, norm_layer=norm_layer,
                                     nonlinearity=nonlinearity, use_spect=opt.use_spect_g, use_coord=opt.use_coord)
            setattr(self, 'mblock' + str(i), block)

        self.down = nn.MaxPool2d(2, stride=2)

        self.mu = nn.Conv2d(opt.ngf * self.mult, opt.ngf * self.mult, 3, stride=1, padding=1)
        self.var = nn.Conv2d(opt.ngf * self.mult, opt.ngf * self.mult, 3, stride=1, padding=1)
    def forward(self, texture, bone):

        x = self.head_0(texture, bone)
        x = self.down(x)
        for i in range(self.layers - 1):
            model = getattr(self, 'down' + str(i))
            x = model(x, bone)
            x = self.down(x)

        for i in range(self.opt.num_blocks):
            model = getattr(self, 'mblock' + str(i))
            x = model(x)
        return self.mu(x), self.var(x)

class ZEncoder(BaseNetwork):
    def __init__(self, opt):
        super(ZEncoder, self).__init__()
        self.opt = opt

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf
        input_nc = opt.image_nc + opt.pose_nc
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)

        self.layer1 = norm_layer(nn.Conv2d(input_nc, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))

        self.actvn = nn.LeakyReLU(0.2, False)

        self.so = s0 = 4
        self.fc_mu = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.fc_var = nn.Linear(ndf * 8 * s0 * s0, 256)
    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.layer1(x)              # 256x256 -> 128x128
        x = self.layer2(self.actvn(x))  # 128x128 -> 64x64
        x = self.layer3(self.actvn(x))  # 64x64 -> 32x32
        x = self.layer4(self.actvn(x))  # 32x32 -> 16x16
        x = self.layer5(self.actvn(x))  # 16x16 -> 8x8
        x = self.layer6(self.actvn(x))  # 8x8 -> 4x4
        x = self.actvn(x)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        z = self.reparameterize(mu, logvar)
        z_dict = {'texture': [mu, logvar]}

        return z, z_dict
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu


class SourceEncoder(nn.Module):
    """
    Source Image Encoder (En_s)
    :param image_nc: number of channels in input image
    :param ngf: base filter channel
    :param img_f: the largest feature channels
    :param encoder_layer: encoder layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param use_spect: use spectual normalization
    :param use_coord: use coordConv operation
    """

    def __init__(self, opt):
        super(SourceEncoder, self).__init__()
        self.opt = opt
        self.encoder_layer = opt.layers_g

        norm_layer = modules.get_norm_layer(norm_type=opt.norm)
        nonlinearity = modules.get_nonlinearity_layer(activation_type=opt.activation)
        input_nc = opt.image_nc

        self.block0 = modules.EncoderBlockOptimized(input_nc, opt.ngf, norm_layer,
                                                    nonlinearity, opt.use_spect_g, opt.use_coord)
        mult = 1
        for i in range(opt.layers_g - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), opt.img_f // opt.ngf)
            block = modules.EncoderBlock(opt.ngf * mult_prev, opt.ngf * mult, norm_layer,
                                         nonlinearity, opt.use_spect_g, opt.use_coord)
            setattr(self, 'encoder' + str(i), block)

        if isinstance(opt.load_size, int) :
            h = w = opt.load_size
        else :
            h, w = opt.load_size
        self.ch = h // 2**(mult-1)
        self.cw = w // 2**(mult-1)
        self.mu = nn.Linear(opt.ngf * mult * self.ch * self.cw, self.ch*self.cw)
        self.var = nn.Linear(opt.ngf * mult * self.ch * self.cw, self.ch*self.cw)

        self.mult = mult
# (ndf * 8 * s0 * s0, 256)
    def forward(self, x):
        out = self.block0(x)
        for i in range(self.encoder_layer - 1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
        out = out.view(x.size(0), -1)
        return self.mu(out), self.var(out)



class DefaultEncoder(BaseNetwork):
    def __init__(self, opt):
        super(DefaultEncoder, self).__init__()
        self.opt = opt
        self.layers = opt.layers_g
        norm_layer = modules.get_norm_layer(norm_type=opt.norm)
        nonlinearity = modules.get_nonlinearity_layer(activation_type=opt.activation)
        input_nc = 2 * opt.pose_nc + opt.image_nc

        self.block0 = modules.EncoderBlockOptimized(input_nc, opt.ngf, norm_layer,
                                                    nonlinearity, opt.use_spect_g, opt.use_coord)
        self.mult = 1
        for i in range(self.layers - 1):
            mult_prev = self.mult
            self.mult = min(2 ** (i + 1), opt.img_f // opt.ngf)
            block = modules.EncoderBlock(opt.ngf * mult_prev, opt.ngf * self.mult, norm_layer,
                                         nonlinearity, opt.use_spect_g, opt.use_coord)
            setattr(self, 'encoder' + str(i), block)

        # ResBlocks
        for i in range(opt.num_blocks):
            block = modules.ResBlock(opt.ngf * self.mult, opt.ngf * self.mult, norm_layer=norm_layer,
                                     nonlinearity=nonlinearity, use_spect=opt.use_spect_g, use_coord=opt.use_coord)
            setattr(self, 'mblock' + str(i), block)

    def forward(self, x, texture_information):
        # Source-to-source Encoder
        x = self.block0(x) # (B, C, H, W) -> (B, ngf, H/2, W/2)
        for i in range(self.layers - 1):
            model = getattr(self, 'encoder' + str(i))
            x = model(x)
        # input_ size : (B, ngf * 2^2, H/2^layers, C/2^layers)
        # Source-to-source Resblocks
        for i in range(self.opt.num_blocks):
            model = getattr(self, 'mblock' + str(i))
            x = model(x)
        return x