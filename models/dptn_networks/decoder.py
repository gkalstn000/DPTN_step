import torch
import torch.nn as nn
from models.dptn_networks import modules


class OutputDncoder(nn.Module):
    def __init__(self, mult, ngf, img_f, layers, norm, activation, use_spect, use_coord, output_nc):
        super(OutputDncoder, self).__init__()
        self.layers = layers
        norm_layer = modules.get_norm_layer(norm_type=norm)
        nonlinearity = modules.get_nonlinearity_layer(activation_type=activation)

        for i in range(self.layers):
            mult_prev = mult
            mult = min(2 ** (self.layers - i - 2), img_f // ngf) if i != self.layers - 1 else 1
            up = modules.ResBlockDecoder(ngf * mult_prev, ngf * mult, ngf * mult, norm_layer,
                                 nonlinearity, use_spect, use_coord)
            setattr(self, 'decoder' + str(i), up)
        self.outconv = modules.Output(ngf, output_nc, 3, None, nonlinearity, use_spect, use_coord)
    def forward(self, x):
        for i in range(self.layers):
            model = getattr(self, 'decoder' + str(i))
            x = model(x)
        out = self.outconv(x)
        return out