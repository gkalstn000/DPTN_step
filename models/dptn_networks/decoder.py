import torch
import torch.nn as nn
from models.dptn_networks import modules
from models.dptn_networks.base_network import BaseNetwork
from models.spade_networks.architecture import SPADEResnetBlock

import torch.nn.functional as F

class SpadeDecoder(BaseNetwork) :
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.layers = opt.layers_g
        nf = opt.ngf
        mult = opt.mult

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        input_nc = 2 * opt.pose_nc + opt.image_nc

        for i in range(self.layers):
            mult_prev = mult
            mult = min(2 ** (self.layers - i - 2), opt.img_f // opt.ngf) if i != self.layers - 1 else 1
            down = SPADEResnetBlock(nf * mult_prev, nf * mult, opt)
            setattr(self, 'decoder' + str(i), down)

        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=4)

    def forward(self, x, texture_information):
        texture_information = torch.cat(texture_information, 1)

        for i in range(self.layers):
            model = getattr(self, 'decoder' + str(i))
            x = model(x, texture_information)
            x = self.up(x)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x
class DefaultDecoder(BaseNetwork):
    def __init__(self, opt):
        super(DefaultDecoder, self).__init__()
        self.opt = opt
        self.layers = opt.layers_g
        mult = opt.mult
        norm_layer = modules.get_norm_layer(norm_type=opt.norm)
        nonlinearity = modules.get_nonlinearity_layer(activation_type=opt.activation)

        for i in range(self.layers):
            mult_prev = mult
            mult = min(2 ** (self.layers - i - 2), opt.img_f // opt.ngf) if i != self.layers - 1 else 1
            up = modules.ResBlockDecoder(opt.ngf * mult_prev, opt.ngf * mult, opt.ngf * mult, norm_layer,
                                 nonlinearity, opt.use_spect_g, opt.use_coord)
            setattr(self, 'decoder' + str(i), up)
        self.outconv = modules.Output(opt.ngf, opt.output_nc, 3, None, nonlinearity, opt.use_spect_g, opt.use_coord)
    def forward(self, x, texture_information):
        for i in range(self.layers):
            model = getattr(self, 'decoder' + str(i))
            x = model(x)
        out = self.outconv(x)
        return out