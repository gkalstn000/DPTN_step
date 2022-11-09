"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import torch.nn as nn

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
        return parser
    def __init__(self, image_nc, pose_nc, ngf=64, img_f=256, layers=3, norm='batch',
                 activation='ReLU', use_spect=True, use_coord=False, output_nc=3, num_blocks=3, affine=True, nhead=2, num_CABs=2, num_TTBs=2):
        super(DPTNGenerator, self).__init__()
        # Encoder En_c
        self.En_c = encoder.InputEncoder(image_nc, pose_nc, ngf, img_f, layers, norm,
                activation, use_spect, use_coord, num_blocks)

        mult = self.En_c.mult

        # Pose Transformer Module (PTM)
        self.PTM = PTM.PoseTransformerModule(d_model=ngf * mult, nhead=nhead, num_CABs=num_CABs,
                                             num_TTBs=num_TTBs, dim_feedforward=ngf * mult,
                                             activation="LeakyReLU", affine=affine, norm=norm)

        # SourceEncoder En_s
        self.En_s = encoder.SourceEncoder(image_nc, ngf, img_f, layers, norm, activation, use_spect, use_coord)
        # OutputDecoder De
        self.De = decoder.OutputDncoder(mult, ngf, img_f, layers, norm,
                 activation, use_spect, use_coord, output_nc)

    def forward(self, source_image, source_bone, target_bone, is_train=True):
        # Self-reconstruction Branch
        # Encode source-to-source
        input_s_s = torch.cat((source_image, source_bone, source_bone), 1)
        F_s_s = self.En_c(input_s_s)
        # Transformation Branch
        # Encode source-to-target
        input_s_t = torch.cat((source_image, source_bone, target_bone), 1)
        F_s_t = self.En_c(input_s_t)
        # Source Image Encoding
        F_s = self.En_s(source_image)
        # Pose Transformer Module for Dual-task Correlation
        F_s_t = self.PTM(F_s_s, F_s_t, F_s)
        # Source-to-source Decoder (only for training)
        out_image_s = None
        if is_train:
            out_image_s = self.De(F_s_s)
        # Source-to-target Decoder
        out_image_t = self.De(F_s_t)
        return out_image_t, out_image_s