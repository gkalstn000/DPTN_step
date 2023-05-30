"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from models.spade_networks.architecture import SPADEResnetBlock, SPAINResnetBlock
import torch
import torch.nn as nn
from models.dptn_networks.base_network import BaseNetwork
from models.dptn_networks.decoder import ImageDecoder
import torch.nn.functional as F
import math

class BaseGenerator(BaseNetwork) :
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--activation', type=str, default='LeakyReLU', help='type of activation function')
        parser.add_argument('--z_dim', type=int, default=256, help="dimension of the latent z vector")
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")


        parser.set_defaults(img_f=512)
        return parser
    def __init__(self):
        super(BaseGenerator, self).__init__()
        self.positional_encoding = PositionalEncoding(1024)
    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / (opt.load_size[1] / opt.load_size[0]))

        return sw, sh
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def get_pose_encoder(self, step):
        return self.positional_encoding(step)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, step):
        return self.pe[step]


class SpadeGenerator(BaseGenerator) :

    def __init__(self, opt):
        super(SpadeGenerator, self).__init__()
        self.opt = opt
        nf = opt.ngf
        norm_nc = opt.pose_nc

        self.sw = self.sh = 32

        self.fc = nn.Linear(opt.z_dim, self.sw * self.sh)

        self.head_0 = SPADEResnetBlock(1, 16 * nf, opt, norm_nc)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt, norm_nc)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt, norm_nc)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt, norm_nc)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt, norm_nc)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt, norm_nc)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt, norm_nc)


        final_nc = 1 * nf
        self.conv_img = nn.Conv2d(final_nc, 1, 3, padding=1)

        self.decoder = ImageDecoder(opt)


    def forward(self, x, pose_information, texture_param, step):
        pe = self.positional_encoding(step).view(-1, 1, self.sh, self.sw)

        if isinstance(x, list) :
            mu, var = x
            z = self.reparameterize(mu, var)
            x = self.fc(z)
            x = x.view(-1, 1, self.sh, self.sw)  # 32, 32

        x = x + pe
        x = self.head_0(x, pose_information)

        x = self.G_middle_0(x, pose_information)
        x = self.G_middle_1(x, pose_information)

        x = self.up_0(x, pose_information)
        x = self.up_1(x, pose_information)
        x = self.up_2(x, pose_information)
        x = self.up_3(x, pose_information)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        img = self.decoder(x)

        return x, img


class SPAINGenerator(BaseGenerator) :
    def __init__(self, opt):
        super(SPAINGenerator, self).__init__()
        self.opt = opt
        nf = opt.ngf
        norm_nc = opt.pose_nc

        self.sw = self.sh = 32

        self.fc = nn.Linear(opt.z_dim, self.sw * self.sh)

        self.head_0 = SPAINResnetBlock(1, 16 * nf, opt, norm_nc)

        self.G_middle_0 = SPAINResnetBlock(16 * nf, 16 * nf, opt, norm_nc)
        self.G_middle_1 = SPAINResnetBlock(16 * nf, 16 * nf, opt, norm_nc)

        self.up_0 = SPAINResnetBlock(16 * nf, 8 * nf, opt, norm_nc)
        self.up_1 = SPAINResnetBlock(8 * nf, 4 * nf, opt, norm_nc)
        self.up_2 = SPAINResnetBlock(4 * nf, 2 * nf, opt, norm_nc)
        self.up_3 = SPAINResnetBlock(2 * nf, 1 * nf, opt, norm_nc)

        final_nc = 1 * nf
        self.conv_img = nn.Conv2d(final_nc, 1, 3, padding=1)

        self.decoder = ImageDecoder(opt)


    def forward(self, x, pose_information, texture_param, step):
        pe = self.positional_encoding(step).view(-1, 1, self.sh, self.sw)
        mu, var = texture_param
        if isinstance(x, list) :
            z = self.reparameterize(mu, var)
            x = self.fc(z)
            x = x.view(-1, 1, self.sh, self.sw)  # 32, 32

        x = x + pe
        x_prev = x

        x = self.head_0(x, pose_information, self.reparameterize(mu, var))

        x = self.G_middle_0(x, pose_information, self.reparameterize(mu, var))
        x = self.G_middle_1(x, pose_information, self.reparameterize(mu, var))

        x = self.up_0(x, pose_information, self.reparameterize(mu, var))
        x = self.up_1(x, pose_information, self.reparameterize(mu, var))
        x = self.up_2(x, pose_information, self.reparameterize(mu, var))
        x = self.up_3(x, pose_information, self.reparameterize(mu, var))

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = x_prev + x
        img = self.decoder(x)

        return x, img

