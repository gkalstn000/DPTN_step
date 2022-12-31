"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 256*256):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
class SPADEAttn(nn.Module):
    def __init__(self, norm_type, norm_nc, texture_information_nc):
        super().__init__()

        if norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)


        nhidden = 128
        # Cross attention part
        self.conv_pose = nn.Conv2d(texture_information_nc, nhidden, kernel_size=3, padding=1)
        self.conv_img = nn.Conv2d(3, nhidden, kernel_size=3, padding=1)
        self.pos_encoder = PositionalEncoding(nhidden)
        self.attn = nn.MultiheadAttention(nhidden, 2)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(nhidden, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, texture_information):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. Multi-head Cross Attention Q: Bone_1, K: Bone_2, V: Image_2
        # texture_information = F.interpolate(texture_information, size=x.size()[2:], mode='nearest')
        q_bone = F.interpolate(texture_information[0], size=x.size()[2:], mode='nearest')
        Q = self.conv_pose(q_bone).flatten(2).permute(2, 0, 1)
        k_bone = F.interpolate(texture_information[1], size=x.size()[2:], mode='nearest')
        K = self.conv_pose(k_bone).flatten(2).permute(2, 0, 1)
        v_img = F.interpolate(texture_information[2], size=x.size()[2:], mode='nearest')
        V = self.conv_img(v_img).flatten(2).permute(2, 0, 1)

        # Part 3. produce scaling and bias conditioned on semantic map
        actv = self.mlp_shared(texture_information)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out



# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
class SPADE(nn.Module):
    def __init__(self, norm_type, norm_nc, texture_information_nc):
        super().__init__()

        if norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(texture_information_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, texture_information):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        texture_information = F.interpolate(texture_information, size=x.size()[2:], mode='nearest')

        actv = self.mlp_shared(texture_information)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

if __name__ == "__main__" :
    config_text = 'spectralspadesyncbatch3x3'
    parsed = re.search('spade(\D+)(\d)x\d', config_text)
    param_free_norm_type = str(parsed.group(1))
    ks = int(parsed.group(2))