import torch
import torch.nn as nn

import models.dptn_networks as networks
import util.util as util
from models.dptn_networks import loss
from collections import defaultdict
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
        src_image, src_map, tgt_image, tgt_map= self.preprocess_input(data)
        if mode == 'generator':

            g_loss, fake_t, fake_s = self.compute_generator_loss(src_image, src_map,
                                                            tgt_image, tgt_map)
            return g_loss, (fake_t, fake_s)
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(src_image, src_map,
                                                     tgt_image, tgt_map)
            return d_loss
        elif mode == 'inference' :
            self.netG.eval()
            with torch.no_grad():
                (gt_tgts, gt_srcs), (fake_tgts, fake_srcs), (vis_tgt, vis_src) = self.generate_fake(src_image, src_map,
                                                                tgt_image,  tgt_map,
                                                                False)
            return (fake_tgts, fake_srcs), (vis_tgt, vis_src)
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


        return data['src_image'], data['src_map'], data['tgt_image'], data['tgt_map']
        # return data['canonical_image'], data['canonical_map'], data['tgt_image'], data['tgt_map'], data['canonical_image'], data['canonical_map']

    def backward_G_basic(self, fake_images, target_images, use_d):
        # Calculate reconstruction loss
        loss_app_gen = 0
        loss_content_gen = 0
        loss_style_gen = 0
        for fake_image, target_image in zip(fake_images, target_images) :
            loss_app_gen += self.L1loss(fake_image, target_image) * self.opt.lambda_rec
            cont, style = self.Vggloss(fake_image, target_image)
            loss_content_gen += cont * self.opt.lambda_content
            loss_style_gen += style * self.opt.lambda_style


        # Calculate GAN loss
        loss_ad_gen = None
        if use_d:
            D_fake = self.netD(fake_image)
            loss_ad_gen = self.GANloss(D_fake, True, False) * self.opt.lambda_g

        return loss_app_gen, loss_ad_gen, loss_style_gen, loss_content_gen
    def compute_generator_loss(self,
                               src_image, src_map,
                               tgt_image, tgt_map):

        G_losses = defaultdict(int)

        (gt_tgts, gt_srcs), (fake_tgts, fake_srcs), (vis_tgt, vis_src) = self.generate_fake(src_image, src_map,
                                                        tgt_image, tgt_map,
                                                        self.opt.isTrain)





        loss_app_gen_t, loss_ad_gen_t, loss_style_gen_t, loss_content_gen_t = self.backward_G_basic(fake_tgts, gt_tgts, use_d=True)
        loss_app_gen_s, _, loss_style_gen_s, loss_content_gen_s = self.backward_G_basic(fake_srcs, gt_srcs, use_d=False)
        self.netD.train()
        G_losses['L1_target'] = self.opt.t_s_ratio * loss_app_gen_t
        G_losses['GAN_target'] = loss_ad_gen_t
        G_losses['VGG_target'] =  self.opt.t_s_ratio * (loss_style_gen_t + loss_content_gen_t)
        G_losses['L1_source'] = (1-self.opt.t_s_ratio) * loss_app_gen_s
        G_losses['VGG_source'] = (1-self.opt.t_s_ratio) * (loss_style_gen_s + loss_content_gen_s)

        return G_losses, vis_tgt, vis_src
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
                                   tgt_image, tgt_map):

        D_losses = {}
        with torch.no_grad():
            _, (fake_tgts, fake_srcs), _ = self.generate_fake(src_image, src_map, tgt_image, tgt_map)

            fake_image_t = fake_tgts[-1].detach()
            fake_image_t.requires_grad_()
            fake_image_s = fake_srcs[-1].detach()
            fake_image_s.requires_grad_()
        self.netG.train()
        D_real_loss, D_fake_loss, gradient_penalty = self.backward_D_basic(tgt_image, fake_image_t)
        D_losses['Real_loss'] = D_real_loss * 0.5
        D_losses['Fake_loss'] = D_fake_loss * 0.5
        if self.opt.gan_mode == 'wgangp':
            D_losses['WGAN_penalty'] = gradient_penalty

        return D_losses

    def generate_fake(self,
                      src_image, src_map,
                      tgt_image, tgt_map,
                      is_train=True):

        b, c, h, w = src_image.size()

        gt_tgts = []
        gt_srcs = []
        fake_tgts = []
        fake_srcs = []

        xt = torch.randn(b, 3, h, w).to(src_image.device)
        for step in range(1, self.opt.step_size + 1) :
            gt_src = self.get_groundtruth(src_image, step)
            gt_tgt = self.get_groundtruth(tgt_image, step)

            xt, xs = self.netG(src_image,
                               src_map,
                               tgt_map,
                               gt_src,
                               xt.detach(),
                               step)

            gt_tgts.append(gt_tgt)
            gt_srcs.append(gt_src)

            fake_tgts.append(xt)
            fake_srcs.append(xs)

        vis_tgt = self.get_vis(gt_tgts, fake_tgts)
        vis_src = self.get_vis(gt_srcs, fake_srcs)



        return (gt_tgts, gt_srcs), (fake_tgts, fake_srcs), (vis_tgt, vis_src)

    def get_vis(self, true_list, fake_list):
        gt_vis = torch.cat(true_list, -1)
        fake_vis = torch.cat(fake_list, -1).detach()
        vis = torch.cat([gt_vis, fake_vis], -2)

        return vis

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

    def get_groundtruth(self, img_tensor, step):
        total_step = self.opt.step_size
        if step == total_step : return img_tensor
        dstep = 255 // total_step
        destory_term = 255 - dstep * step

        img_tensor_denorm = (img_tensor + 1) / 2 * 255

        ground_truth = img_tensor_denorm // destory_term * destory_term

        ground_truth = (ground_truth / 255 * 2) - 1

        return ground_truth