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
            self.GANloss = loss.GANLoss(opt.gan_mode,
                                        tensor=self.FloatTensor, opt=self.opt).cuda()
            self.L1loss = torch.nn.L1Loss()
            self.Vggloss = loss.VGGLoss().cuda()
            self.L2loss = torch.nn.MSELoss()
            self.KLDLoss = loss.KLDLoss()


    def forward(self, data, mode):
        src_image, src_map, tgt_image, tgt_map = self.preprocess_input(data)
        if mode == 'generator':

            g_loss, fake_t = self.compute_generator_loss(src_image, src_map,
                                                            tgt_image, tgt_map)
            return g_loss, fake_t
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(src_image, src_map,
                                                     tgt_image, tgt_map)
            return d_loss
        elif mode == 'inference' :
            self.netG.eval()
            with torch.no_grad():
                fake_image_t, z_dict = self.generate_fake(src_image, src_map,
                                                                  tgt_map,
                                                                False)
            return fake_image_t
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
            data['src_image'] = data['src_image'].float().cuda()
            data['src_map'] = data['src_map'].float().cuda()
            data['tgt_image'] = data['tgt_image'].float().cuda()
            data['tgt_map'] = data['tgt_map'].float().cuda()

        return data['src_image'], data['src_map'], data['tgt_image'], data['tgt_map']

    def backward_G_basic(self, fake_image, target_image, use_d):
        # Calculate reconstruction loss
        loss_app_gen = self.L1loss(fake_image, target_image)
        loss_app_gen = loss_app_gen * self.opt.lambda_rec

        # Calculate perceptual loss
        loss_content_gen, loss_style_gen = self.Vggloss(fake_image, target_image)
        loss_style_gen = loss_style_gen * self.opt.lambda_style
        loss_content_gen = loss_content_gen * self.opt.lambda_content

        return loss_app_gen, loss_style_gen, loss_content_gen
    def compute_generator_loss(self,
                               src_image, src_map,
                               tgt_image, tgt_map):
        self.netD.eval()
        self.netG.train()
        G_losses = {}

        fake_image_t, z_dict = self.generate_fake(src_image, src_map, tgt_map, self.opt.isTrain)

        pred_fake, pred_real = self.backward_D_basic(tgt_map, fake_image_t, tgt_image)

        loss_app_gen_t, loss_style_gen_t, loss_content_gen_t = self.backward_G_basic(fake_image_t, tgt_image, use_d=True)

        self.netD.train()

        # G_losses['L1_cycle'] = self.opt.t_s_ratio * self.L1loss(fake_image_s_cycle, src_image) * self.opt.lambda_rec
        G_losses['GAN'] = self.GANloss(pred_fake, True, for_discriminator=False)
        G_losses['VGG_target'] =  self.opt.t_s_ratio * (loss_style_gen_t + loss_content_gen_t)
        G_losses['KLD_texture_loss'] = self.KLDLoss(z_dict['texture']) * self.opt.lambda_kld

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.L1loss(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss


        return G_losses, fake_image_t
    def backward_D_basic(self, bone_map, fake, real):
        fake_concat = torch.cat([bone_map, fake], dim=1)
        real_concat = torch.cat([bone_map, real], dim=1)
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)
        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real
    def compute_discriminator_loss(self,
                                   src_image, src_map,
                                   tgt_image, tgt_map):
        D_losses = {}
        with torch.no_grad():
            fake_image_t, _ = self.generate_fake(src_image, src_map, tgt_map)
            fake_image_t = fake_image_t.detach()
            fake_image_t.requires_grad_()

        pred_fake, pred_real = self.backward_D_basic(tgt_map, fake_image_t, tgt_image)

        D_losses['D_fake'] = self.GANloss(pred_fake, False, for_discriminator=True)
        D_losses['D_real'] = self.GANloss(pred_real, True, for_discriminator=True)

        return D_losses

    def generate_fake(self,
                      src_image, src_map,
                      tgt_map,
                      is_train=True):

        fake_image_t, z_dict = self.netG(src_image, src_map, tgt_map,
                                                      is_train)

        return fake_image_t, z_dict
    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real