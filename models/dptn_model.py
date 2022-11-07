import numpy as np
import torch
import os
import itertools
from torch.autograd import Variable
from models.networks.dptn.base_model import BaseModel
from models.networks.dptn import networks
from models.networks.dptn import external_function
from models.networks.dptn import base_function
import util.util as util


class DPTNModel(BaseModel):
    def name(self):
        return 'DPTNModel'

    @staticmethod
    def modify_options(parser, is_train=True):
        """Add new options and rewrite default values for existing options"""
        parser.add_argument('--init_type', type=str, default='orthogonal', help='initial type')
        parser.add_argument('--use_spect_g', action='store_false', help='use spectual normalization in generator')
        parser.add_argument('--use_spect_d', action='store_false', help='use spectual normalization in generator')
        parser.add_argument('--use_coord', action='store_true', help='use coordconv')
        parser.add_argument('--lambda_style', type=float, default=500, help='weight for the VGG19 style loss')
        parser.add_argument('--lambda_content', type=float, default=0.5, help='weight for the VGG19 content loss')
        parser.add_argument('--layers_g', type=int, default=3, help='number of layers in G')
        parser.add_argument('--save_input', action='store_true', help="whether save the input images when testing")
        parser.add_argument('--num_blocks', type=int, default=3, help="number of resblocks")
        parser.add_argument('--affine', action='store_true', default=True, help="affine in PTM")
        parser.add_argument('--nhead', type=int, default=2, help="number of heads in PTM")
        parser.add_argument('--num_CABs', type=int, default=2, help="number of CABs in PTM")
        parser.add_argument('--num_TTBs', type=int, default=2, help="number of CABs in PTM")

        # if is_train:
        parser.add_argument('--ratio_g2d', type=float, default=0.1, help='learning rate ratio G to D')
        parser.add_argument('--lambda_rec', type=float, default=5.0, help='weight for image reconstruction loss')
        parser.add_argument('--lambda_g', type=float, default=2.0, help='weight for generation loss')
        parser.add_argument('--t_s_ratio', type=float, default=0.5, help='loss ratio between dual tasks')
        parser.add_argument('--dis_layers', type=int, default=4, help='number of layers in D')
        parser.set_defaults(use_spect_g=False)
        parser.set_defaults(use_spect_d=True)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.opt = opt
        self.old_size = opt.old_size
        self.t_s_ratio = opt.t_s_ratio
        self.loss_names = ['app_gen_s', 'content_gen_s', 'style_gen_s', 'app_gen_t', 'ad_gen_t', 'dis_img_gen_t', 'content_gen_t', 'style_gen_t']
        self.model_names = ['G']
        self.visual_names = ['source_image', 'source_pose', 'target_image', 'target_pose', 'fake_image_s', 'fake_image_t']

        self.net_G = networks.define_G(opt, image_nc=opt.image_nc, pose_nc=opt.structure_nc, ngf=64, img_f=512,
                                       encoder_layer=3, norm=opt.norm, activation='LeakyReLU',
                                       use_spect=opt.use_spect_g, use_coord=opt.use_coord, output_nc=3, num_blocks=3, affine=True, nhead=opt.nhead, num_CABs=opt.num_CABs, num_TTBs=opt.num_TTBs)

        # Discriminator network
        if self.isTrain:
            self.model_names = ['G', 'D']
            self.net_D = networks.define_D(opt, ndf=32, img_f=128, layers=opt.dis_layers, use_spect=opt.use_spect_d)

        if self.opt.verbose:
                print('---------- Networks initialized -------------')
        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.old_lr = opt.lr

            self.GANloss = external_function.GANLoss(opt.gan_mode).to(opt.device)
            self.L1loss = torch.nn.L1Loss()
            self.Vggloss = external_function.VGGLoss().to(opt.device)
            # self.schedulers = [base_function.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        else:
            self.net_G.eval()

        if not self.isTrain or opt.continue_train:
            print('model resumed from latest')
            netG = util.load_network(self.net_G, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(self.net_D, 'D', opt.which_epoch, opt)


    def create_optimizers(self, opt):
        # define the optimizer
        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(itertools.chain(
            filter(lambda p: p.requires_grad, self.net_G.parameters())),
            lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(itertools.chain(
            filter(lambda p: p.requires_grad, self.net_D.parameters())),
            lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def preprocess_input(self, data):
        # move to GPU and change data types
        if self.use_gpu():
            data['src_image'] = data['src_image'].cuda()
            data['src_bone'] = data['src_bone'].cuda()
            data['tgt_image'] = data['tgt_image'].cuda()
            data['tgt_bone'] = data['tgt_bone'].cuda()
            data['canonical_image'] = data['canonical_image'].cuda()
            data['canonical_bone'] = data['canonical_bone'].cuda()
        return data['src_image'], data['src_bone'], data['tgt_image'], data['tgt_bone'], data['canonical_image'], data['canonical_bone']

    def generate_fake(self, src_image, src_bone, tgt_bone):
        self.fake_image_t, self.fake_image_s = self.net_G(src_image, src_bone, tgt_bone)

        return self.fake_image_t, self.fake_image_s




    def discriminate(self, netD, real, fake):
        D_real = netD(real)
        D_fake = netD(fake.detach())

        return D_fake, D_real

    def compute_discriminator_loss(self, src_image, src_bone, tgt_image, tgt_bone):
        base_function._unfreeze(self.net_D)
        D_losses = {}
        with torch.no_grad():
            fake_image = self.generate_fake(src_image, src_bone, tgt_bone)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()
        pred_fake, pred_real = self.discriminate(tgt_bone, fake_image, tgt_image)

        D_real_loss = self.GANloss(pred_real, True, True)
        D_fake_loss = self.GANloss(pred_fake, False, True)
        D_loss = (D_real_loss + D_fake_loss) * 0.5
        # gradient penalty for wgan-gp
        if self.opt.gan_mode == 'wgangp':
            gradient_penalty, gradients = external_function.cal_gradient_penalty(self.net_D, tgt_image, fake_image)
            D_loss += gradient_penalty

        D_losses['D_loss'] = D_loss
        return D_loss

    def compute_generator_loss(self, src_image, src_bone, tgt_image, tgt_bone):
        G_losses = {}
        fake_image = self.generate_fake(src_image, src_bone, tgt_bone)

        base_function._freeze(self.net_D)
        pred_fake, pred_real = self.discriminate(tgt_bone, fake_image, tgt_image)
        G_losses['GAN'] = self.GANloss(pred_fake, True, False) * self.opt.lambda_g

        # Calculate reconstruction loss
        loss_app_gen = self.L1loss(fake_image, tgt_image)
        G_losses['L1'] = loss_app_gen * self.opt.lambda_rec

        # Calculate perceptual loss
        loss_content_gen, loss_style_gen = self.Vggloss(fake_image, tgt_image)
        G_losses['Style'] = loss_style_gen * self.opt.lambda_style
        G_losses['Content'] = loss_content_gen * self.opt.lambda_content

        return G_losses, fake_image

    def backward_G(self):
        base_function._unfreeze(self.net_D)

        self.loss_app_gen_t, self.loss_ad_gen_t, self.loss_style_gen_t, self.loss_content_gen_t = self.backward_G_basic(self.fake_image_t, self.target_image, use_d = True)

        self.loss_app_gen_s, self.loss_ad_gen_s, self.loss_style_gen_s, self.loss_content_gen_s = self.backward_G_basic(self.fake_image_s, self.source_image, use_d = False)
        G_loss = self.t_s_ratio*(self.loss_app_gen_t+self.loss_style_gen_t+self.loss_content_gen_t) + (1-self.t_s_ratio)*(self.loss_app_gen_s+self.loss_style_gen_s+self.loss_content_gen_s)+self.loss_ad_gen_t
        G_loss.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0