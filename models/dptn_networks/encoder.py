import torch
import torch.nn as nn
from models.dptn_networks import modules


class InputEncoder(nn.Module):
    def __init__(self, image_nc, pose_nc, ngf, img_f, layers, norm, activation, use_spect, use_coord, num_blocks):
        super(InputEncoder, self).__init__()
        self.layers = layers
        norm_layer = modules.get_norm_layer(norm_type=norm)
        nonlinearity = modules.get_nonlinearity_layer(activation_type=activation)
        input_nc = 2 * pose_nc + image_nc

        self.block0 = modules.EncoderBlockOptimized(input_nc, ngf, norm_layer,
                                                    nonlinearity, use_spect, use_coord)
        self.mult = 1
        for i in range(self.layers - 1):
            mult_prev = self.mult
            self.mult = min(2 ** (i + 1), img_f // ngf)
            block = modules.EncoderBlock(ngf * mult_prev, ngf * self.mult, norm_layer,
                                         nonlinearity, use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)

        # ResBlocks
        self.num_blocks = num_blocks
        for i in range(num_blocks):
            block = modules.ResBlock(ngf * self.mult, ngf * self.mult, norm_layer=norm_layer,
                                     nonlinearity=nonlinearity, use_spect=use_spect, use_coord=use_coord)
            setattr(self, 'mblock' + str(i), block)

    def forward(self, x):
        # Source-to-source Encoder
        F_s_s = self.block0(x)
        for i in range(self.layers - 1):
            model = getattr(self, 'encoder' + str(i))
            F_s_s = model(F_s_s)
        # Source-to-source Resblocks
        for i in range(self.num_blocks):
            model = getattr(self, 'mblock' + str(i))
            F_s_s = model(F_s_s)
        return F_s_s


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

    def __init__(self, image_nc, ngf=64, img_f=1024, encoder_layer=3, norm='batch',
                 activation='ReLU', use_spect=True, use_coord=False):
        super(SourceEncoder, self).__init__()

        self.encoder_layer = encoder_layer

        norm_layer = modules.get_norm_layer(norm_type=norm)
        nonlinearity = modules.get_nonlinearity_layer(activation_type=activation)
        input_nc = image_nc

        self.block0 = modules.EncoderBlockOptimized(input_nc, ngf, norm_layer,
                                                    nonlinearity, use_spect, use_coord)
        mult = 1
        for i in range(encoder_layer - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ngf)
            block = modules.EncoderBlock(ngf * mult_prev, ngf * mult, norm_layer,
                                         nonlinearity, use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)

    def forward(self, x):
        out = self.block0(x)
        for i in range(self.encoder_layer - 1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
        return out
