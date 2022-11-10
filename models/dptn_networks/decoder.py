import torch
import torch.nn as nn
from models.dptn_networks import modules


class OutputDncoder(nn.Module):
    def __init__(self, mult, opt):
        super(OutputDncoder, self).__init__()
        self.opt = opt
        self.layers = opt.layers_g
        norm_layer = modules.get_norm_layer(norm_type=opt.norm)
        nonlinearity = modules.get_nonlinearity_layer(activation_type=opt.activation)

        for i in range(self.layers):
            mult_prev = mult
            mult = min(2 ** (self.layers - i - 2), opt.img_f // opt.ngf) if i != self.layers - 1 else 1
            up = modules.ResBlockDecoder(opt.ngf * mult_prev, opt.ngf * mult, opt.ngf * mult, norm_layer,
                                 nonlinearity, opt.use_spect_g, opt.use_coord)
            setattr(self, 'decoder' + str(i), up)
        self.outconv = modules.Output(opt.ngf, opt.output_nc, 3, None, nonlinearity, opt.use_spect_g, opt.use_coord)
    def forward(self, x):
        for i in range(self.layers):
            model = getattr(self, 'decoder' + str(i))
            x = model(x)
        out = self.outconv(x)
        return out