"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import torch.nn as nn

from models.dptn_networks import define_En_c, define_De
from models.dptn_networks.base_network import BaseNetwork
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
        parser.add_argument('--type_En_c', type=str, default='default', help='selects En_c type to use for generator (default | spade | spadeattn)')
        parser.add_argument('--type_Dc', type=str, default='default', help='selects Dc type to use for generator (default | spade | spadeattn)')

        parser.set_defaults(use_spect_g=True)
        parser.set_defaults(use_coord=False)
        parser.set_defaults(norm='instance')
        parser.set_defaults(img_f=512)
        return parser
    def __init__(self, opt):
        super(DPTNGenerator, self).__init__()
        self.opt = opt
        # Encoder En_c
        # self.En_c = encoder.DefaultEncoder(opt)
        self.En_c = define_En_c(opt)
        opt.mult = self.En_c.mult
        # Pose Transformer Module (PTM)
        self.PTM = PTM.PoseTransformerModule(opt=opt)
        # SourceEncoder En_s
        self.En_s = encoder.SourceEncoder(opt)
        # OutputDecoder De
        self.De = define_De(opt)

    def forward(self, source_image, source_bone, target_bone,
                canonical_image, canonical_bone, is_train=True):
        # Encode source-to-source
        input_s_s = torch.cat((source_image, source_bone, source_bone), 1)
        texture_information = [source_bone, source_image] # canonical->canonical information
        F_s_s = self.En_c(input_s_s, texture_information)
        # Encode source-to-target
        input_s_t = torch.cat((source_image, source_bone, target_bone), 1)
        texture_information = [source_bone, source_image]  # source-canonical information
        F_s_t = self.En_c(input_s_t, texture_information)
        # Source Image Encoding
        F_s = self.En_s(source_image)
        # Pose Transformer Module for Dual-task Correlation
        F_s_t, first_attn_weights, last_attn_weights = self.PTM(F_s_s, F_s_t, F_s)
        # Source-to-source Decoder (only for training)
        out_image_s = None
        if is_train:
            texture_information = [source_bone, source_image] # [canonical_bone, source_bone, source_image]
            out_image_s = self.De(F_s_s, texture_information)
        # Source-to-target Decoder
        texture_information = [source_bone, source_image] # [target_bone, source_bone, source_image]
        out_image_t = self.De(F_s_t, texture_information)
        return out_image_t, out_image_s, first_attn_weights, last_attn_weights