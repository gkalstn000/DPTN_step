import torch
import torch.nn as nn

import models.dptn_networks as networks
import util.util as util
from models.dptn_networks import loss

class DPTNModel(nn.Module) :
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser
    def __init__(self, opt):
        super(DPTNModel, self).__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG, self.netD = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.GANloss = loss.GANLoss(opt.gan_mode).cuda()
            self.L1loss = torch.nn.L1Loss()
            self.Vggloss = loss.VGGLoss().cuda()


    def forward(self, data, mode):
        src_image, src_map, tgt_image, tgt_map, can_image, can_map = self.preprocess_input(data)
        if mode == 'generator':
            g_loss, fake_t, fake_s = self.compute_generator_loss(src_image, src_map,
                                                            tgt_image, tgt_map,
                                                            can_image, can_map)
            return g_loss, fake_t, fake_s
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(src_image, src_map,
                                                     tgt_image, tgt_map,
                                                     can_image, can_map)
            return d_loss
    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        D_params = list(self.netD.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)


    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None
        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
        return netG, netD
    def preprocess_input(self, data):
        if self.use_gpu():
            # data['src_image'] = data['src_image'].to(f'cuda:{self.opt.gpu_ids[0]}')
            # data['src_map'] = data['src_map'].to(f'cuda:{self.opt.gpu_ids[0]}')
            # data['tgt_image'] = data['tgt_image'].to(f'cuda:{self.opt.gpu_ids[0]}')
            # data['tgt_map'] = data['tgt_map'].to(f'cuda:{self.opt.gpu_ids[0]}')
            # data['canonical_image'] = data['canonical_image'].to(f'cuda:{self.opt.gpu_ids[0]}')
            # data['canonical_map'] = data['canonical_map'].to(f'cuda:{self.opt.gpu_ids[0]}')
            data['src_image'] = data['src_image'].float().cuda()
            data['src_map'] = data['src_map'].float().cuda()
            data['tgt_image'] = data['tgt_image'].float().cuda()
            data['tgt_map'] = data['tgt_map'].float().cuda()
            data['canonical_image'] = data['canonical_image'].float().cuda()
            data['canonical_map'] = data['canonical_map'].float().cuda()

        # return data['src_image'], data['src_map'], data['tgt_image'], data['tgt_map'], data['canonical_image'], data['canonical_map']
        return data['canonical_image'], data['canonical_map'], data['tgt_image'], data['tgt_map'], data['canonical_image'], data['canonical_map']

    def backward_G_basic(self, fake_image, target_image, use_d):
        # Calculate reconstruction loss
        loss_app_gen = self.L1loss(fake_image, target_image)
        loss_app_gen = loss_app_gen * self.opt.lambda_rec

        # Calculate GAN loss
        loss_ad_gen = None
        if use_d:
            self.netD.eval()
            with torch.no_grad():
                D_fake = self.netD(fake_image)
            loss_ad_gen = self.GANloss(D_fake, True, False) * self.opt.lambda_g

        # Calculate perceptual loss
        loss_content_gen, loss_style_gen = self.Vggloss(fake_image, target_image)
        loss_style_gen = loss_style_gen * self.opt.lambda_style
        loss_content_gen = loss_content_gen * self.opt.lambda_content

        return loss_app_gen, loss_ad_gen, loss_style_gen, loss_content_gen
    def compute_generator_loss(self,
                               src_image, src_map,
                               tgt_image, tgt_map,
                               can_image, can_map):
        self.netD.train()
        G_losses = {}
        fake_image_t, fake_image_s = self.generate_fake(src_image, src_map,
                                                                  tgt_map,
                                                                  can_image, can_map)
        loss_app_gen_t, loss_ad_gen_t, loss_style_gen_t, loss_content_gen_t = self.backward_G_basic(fake_image_t, tgt_image, use_d=True)
        loss_app_gen_s, _, loss_style_gen_s, loss_content_gen_s = self.backward_G_basic(fake_image_s, src_image, use_d=False)
        G_losses['L1_target'] = self.opt.t_s_ratio * loss_app_gen_t
        G_losses['GAN_target'] = loss_ad_gen_t
        G_losses['VGG_target'] =  self.opt.t_s_ratio * (loss_style_gen_t + loss_content_gen_t)
        G_losses['L1_source'] = (1-self.opt.t_s_ratio) * loss_app_gen_s
        G_losses['VGG_source'] = (1-self.opt.t_s_ratio) * (loss_style_gen_s + loss_content_gen_s)

        return G_losses, fake_image_t, fake_image_s
    def backward_D_basic(self, real, fake):
        # Real
        D_real = self.netD(real)
        D_real_loss = self.GANloss(D_real, True, True)
        # fake
        D_fake = self.netD(fake.detach())
        D_fake_loss = self.GANloss(D_fake, False, True)
        # gradient penalty for wgan-gp
        gradient_penalty = 0
        if self.opt.gan_mode == 'wgangp':
            gradient_penalty, gradients = loss.cal_gradient_penalty(self.netD, real, fake.detach())

        return D_real_loss, D_fake_loss, gradient_penalty
    def compute_discriminator_loss(self,
                                   src_image, src_map,
                                   tgt_image, tgt_map,
                                   can_image, can_map):
        self.netD.train()
        D_losses = {}
        with torch.no_grad():
            fake_image_t, fake_image_s = self.netG(src_image, src_map,
                                                   tgt_map,
                                                   can_image, can_map)
            fake_image_t = fake_image_t.detach()
            fake_image_t.requires_grad_()
            fake_image_s = fake_image_s.detach()
            fake_image_s.requires_grad_()

        D_real_loss, D_fake_loss, gradient_penalty = self.backward_D_basic(tgt_image, fake_image_t)
        D_losses['Real_loss'] = D_real_loss * 0.5
        D_losses['Fake_loss'] = D_fake_loss * 0.5
        D_losses['WGAN_penalty'] = gradient_penalty

        return D_losses

    def generate_fake(self,
                      src_image, src_map,
                      tgt_map,
                      can_image, can_map):

        fake_image_t, fake_image_s = self.netG(src_image, src_map,
                                               tgt_map,
                                               can_image, can_map)
        return fake_image_t, fake_image_s
    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0