import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dptn_networks import modules
from models.dptn_networks.base_network import BaseNetwork
from models.spade_networks.architecture import SPADEResnetBlock

class SpadeAttnEncoder(BaseNetwork) :
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.layers = opt.layers_g
        nf = opt.ngf
        nonlinearity = modules.get_nonlinearity_layer(activation_type=opt.activation)
        norm_layer = modules.get_norm_layer(norm_type=opt.norm)

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        input_nc = 2 * opt.pose_nc + opt.image_nc

        self.head_0 = SPADEResnetBlock(input_nc, nf, opt, 'encoder')

        self.mult = 1
        for i in range(self.layers - 1) :
            mult_prev = self.mult
            self.mult = min(2 ** (i + 1), opt.img_f // opt.ngf)
            block = SPADEResnetBlock(opt.ngf * mult_prev, opt.ngf * self.mult, opt, 'encoder')
            setattr(self, 'down' + str(i), block)

        # ResBlocks
        for i in range(opt.num_blocks):
            block = modules.ResBlock(opt.ngf * self.mult, opt.ngf * self.mult, norm_layer=norm_layer,
                                     nonlinearity=nonlinearity, use_spect=opt.use_spect_g, use_coord=opt.use_coord)
            setattr(self, 'mblock' + str(i), block)

        self.down = nn.MaxPool2d(2, stride=2)
    def forward(self, x, texture_information):
        x = self.head_0(x, texture_information)
        x = self.down(x)
        for i in range(self.layers - 1):
            model = getattr(self, 'down' + str(i))
            x = model(x, texture_information)
            x = self.down(x)

        for i in range(self.opt.num_blocks):
            model = getattr(self, 'mblock' + str(i))
            x = model(x)
        return x

class SpadeEncoder(BaseNetwork) :
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.layers = opt.layers_g
        nf = opt.ngf
        nonlinearity = modules.get_nonlinearity_layer(activation_type=opt.activation)
        norm_layer = modules.get_norm_layer(norm_type=opt.norm)

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        input_nc = 2 * opt.pose_nc + opt.image_nc

        self.head_0 = SPADEResnetBlock(input_nc, nf, opt, 'encoder')

        self.mult = 1
        for i in range(self.layers - 1) :
            mult_prev = self.mult
            self.mult = min(2 ** (i + 1), opt.img_f // opt.ngf)
            block = SPADEResnetBlock(opt.ngf * mult_prev, opt.ngf * self.mult, opt, 'encoder')
            setattr(self, 'down' + str(i), block)

        # ResBlocks
        for i in range(opt.num_blocks):
            block = modules.ResBlock(opt.ngf * self.mult, opt.ngf * self.mult, norm_layer=norm_layer,
                                     nonlinearity=nonlinearity, use_spect=opt.use_spect_g, use_coord=opt.use_coord)
            setattr(self, 'mblock' + str(i), block)

        self.down = nn.MaxPool2d(2, stride=2)
    def forward(self, x, texture_information):
        texture_information = torch.cat(texture_information, 1)

        x = self.head_0(x, texture_information)
        x = self.down(x)
        for i in range(self.layers - 1):
            model = getattr(self, 'down' + str(i))
            x = model(x, texture_information)
            x = self.down(x)

        for i in range(self.opt.num_blocks):
            model = getattr(self, 'mblock' + str(i))
            x = model(x)
        return x

class AttnEncoder(BaseNetwork):
    def __init__(self, opt):
        super(AttnEncoder, self).__init__()
        self.opt = opt
        self.layers = opt.layers_g
        norm_layer = modules.get_norm_layer(norm_type=opt.norm)
        nonlinearity = modules.get_nonlinearity_layer(activation_type=opt.activation)

        self.query_encoder = nn.ModuleList()
        self.key_encoder = nn.ModuleList()
        self.value_encoder = nn.ModuleList()

        in_channel_query = opt.pose_nc
        in_channel_key = opt.pose_nc + opt.image_nc
        in_channel_value = opt.image_nc
        out_channel = opt.ngf
        for i in range(self.layers):
            self.query_encoder.append(modules.EncoderBlock(in_channel_query, out_channel, norm_layer,
                                         nonlinearity, opt.use_spect_g, opt.use_coord))
            self.key_encoder.append(modules.EncoderBlock(in_channel_key, out_channel, norm_layer,
                                         nonlinearity, opt.use_spect_g, opt.use_coord))
            self.value_encoder.append(modules.EncoderBlock(in_channel_value, out_channel, norm_layer,
                                                         nonlinearity, opt.use_spect_g, opt.use_coord))
            self.mult = min(2 ** (i + 1), opt.img_f // opt.ngf)

            in_channel_query = out_channel
            in_channel_key = out_channel
            in_channel_value = out_channel
            out_channel = opt.ngf * self.mult
        self.mult //= 2

        self.query_encoder = nn.Sequential(*self.query_encoder)
        self.key_encoder = nn.Sequential(*self.key_encoder)
        self.value_encoder = nn.Sequential(*self.value_encoder)

        self.Attn = modules.CrossAttnModule(opt.ngf * self.mult, opt.nhead, opt.ngf * self.mult)

        # ResBlocks
        for i in range(opt.num_blocks):
            block = modules.ResBlock(opt.ngf * self.mult, opt.ngf * self.mult, norm_layer=norm_layer,
                                     nonlinearity=nonlinearity, use_spect=opt.use_spect_g, use_coord=opt.use_coord)
            setattr(self, 'mblock' + str(i), block)


    def forward(self, target_bone, source_bone, source_image, train_mode = True):
        '''
        :param Query: Query Bone
        :param Key: Source Bone
        :param Value: Source Image
        :return:
        '''
        F_S_S_query = self.query_encoder(source_bone)
        key = self.key_encoder(torch.cat([source_bone, source_image], 1))
        value = self.value_encoder(source_image)
        F_S_S, _ = self.Attn(F_S_S_query, key, value)

        F_S_T = None
        if train_mode:
            F_S_T_query = self.query_encoder(target_bone)
            F_S_T, _ = self.Attn(F_S_T_query, key, value)

        # Source-to-source Resblocks
        for i in range(self.opt.num_blocks):
            model = getattr(self, 'mblock' + str(i))
            F_S_S = model(F_S_S)
            if train_mode:
                F_S_T = model(F_S_S)
        return F_S_S, F_S_T

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

    def forward(self, x):
        out = self.block0(x)
        for i in range(self.encoder_layer - 1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
        return out
