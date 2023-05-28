import torch
import torch.nn as nn
from models.dptn_networks import modules
from models.dptn_networks.base_network import BaseNetwork
from models.spade_networks.architecture import SPADEResnetBlock, SPAINResnetBlock

import torch.nn.functional as F

class ImageDecoder(nn.Module) :
    def __init__(self, opt):
        super(ImageDecoder, self).__init__()
        self.opt = opt
        nf = opt.ngf
        norm_nc = opt.pose_nc

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt, norm_nc)
        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt, norm_nc)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt, norm_nc)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt, norm_nc)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt, norm_nc)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt, norm_nc)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt, norm_nc)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt, norm_nc)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)


    def forward(self, texture_param, pose_information):
        mu, var = texture_param
        z = self.reparameterize(mu, var)

        x = self.fc(z)
        x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)

        x = self.head_0(x, pose_information)

        x = self.up(x)
        x = self.G_middle_0(x, pose_information)
        x = self.G_middle_1(x, pose_information)

        x = self.up(x)
        x = self.up_0(x, pose_information)
        x = self.up(x)
        x = self.up_1(x, pose_information)
        x = self.up(x)
        x = self.up_2(x, pose_information)
        x = self.up(x)
        x = self.up_3(x, pose_information)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x
