"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import torch.nn as nn

from models.dptn_networks import define_En_c, define_De
from models.dptn_networks.base_network import BaseNetwork
from models.spade_networks.architecture import SPADEResnetBlock
import torch.nn.functional as F

from models.dptn_networks import encoder
from models.dptn_networks import decoder
from models.dptn_networks import PTM

class DPTNGenerator(BaseNetwork):
    """
    Dual-task Pose Transformer Network (DPTN)
    :param image_nc: number of channels in input image
    :param pose_nc: number of channels in input pose
    :param ngf: base filter channel
    :param img_f: the largest feature channels
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param use_spect: use spectual normalization
    :param use_coord: use coordConv operation
    :param output_nc: number of channels in output image
    :param num_blocks: number of ResBlocks
    :param affine: affine in Pose Transformer Module
    :param nhead: number of heads in attention module
    :param num_CABs: number of CABs
    :param num_TTBs: number of TTBs
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--activation', type=str, default='LeakyReLU', help='type of activation function')
        parser.add_argument('--type_En_c', type=str, default='default', help='selects En_c type to use for generator (z | dptn)')
        parser.add_argument('--type_Dc', type=str, default='default', help='selects Dc type to use for generator (default | spade)')
        parser.add_argument('--z_dim', type=int, default=256, help="dimension of the latent z vector")
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")


        parser.set_defaults(img_f=512)
        return parser
    def __init__(self, opt):
        super(DPTNGenerator, self).__init__()
        self.opt = opt
        self.z_encoder = define_En_c(opt)
        self.decoder = define_De(opt)
    def forward(self, texture, bone):
        encoder_input = torch.cat([texture], 1)
        z, z_dict = self.z_encoder(encoder_input)

        external_information = [bone]
        x = self.decoder(z, external_information)

        return x, z_dict


