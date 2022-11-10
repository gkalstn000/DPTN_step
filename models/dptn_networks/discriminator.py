import torch.nn as nn
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm

from models.dptn_networks.base_network import BaseNetwork
from models.dptn_networks import modules

class ResDiscriminator(BaseNetwork):
    """
    ResNet Discriminator Network
    :param input_nc: number of channels in input
    :param ndf: base filter channel
    :param layers: down and up sample layers
    :param img_f: the largest feature channels
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param use_spect: use spectual normalization
    :param use_coord: use coordConv operation
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(use_spect_d=True)
        return parser

    def __init__(self, opt, input_nc=3, ndf=32, img_f=128, norm='none', activation='LeakyReLU', use_spect=True,
                 use_coord=False):
        super(ResDiscriminator, self).__init__()
        self.opt = opt
        self.layers = opt.dis_layers
        norm_layer = modules.get_norm_layer(norm_type=norm)
        nonlinearity = modules.get_nonlinearity_layer(activation_type=activation)
        self.nonlinearity = nonlinearity

        # encoder part
        self.block0 = modules.ResBlockEncoderOptimized(input_nc, ndf, ndf, norm_layer, nonlinearity, opt.use_spect_d, use_coord)

        mult = 1
        for i in range(opt.dis_layers - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f//ndf)
            block = modules.ResBlockEncoder(ndf*mult_prev, ndf*mult, ndf*mult_prev, norm_layer, nonlinearity, opt.use_spect_d, use_coord)
            setattr(self, 'encoder' + str(i), block)
        self.conv = SpectralNorm(nn.Conv2d(ndf*mult, 1, 1))

    def forward(self, x):
        out = self.block0(x)
        for i in range(self.layers - 1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
        out = self.conv(self.nonlinearity(out))
        return out