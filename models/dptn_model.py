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

        self.netG, self.netE, self.netD = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.GANloss = loss.GANLoss(opt.gan_mode, tensor=self.FloatTensor, opt=self.opt).cuda()
            self.L1loss = torch.nn.L1Loss()
            self.Vggloss = loss.VGGLoss()
            self.L2loss = torch.nn.MSELoss()
            self.KLDLoss = loss.KLDLoss()


    def forward(self, data, mode, flag):
        P1, B2, P2 = self.preprocess_input(data, flag)
        if mode == 'generator':
            g_loss, fake_t = self.compute_generator_loss(P1, B2, P2)
            return g_loss, fake_t
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(P1, B2, P2)
            return d_loss
        elif mode == 'inference' :
            self.netG.eval()
            self.netE.eval()
            with torch.no_grad():
                mu, logvar, _ = self.encoding(P1, loss=False)
                texture_param = [mu, logvar]
                fake_image_t, _, _, vis = self.generate_fake(texture_param, B2, P2)
            return fake_image_t
    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters()) + list(self.netE.parameters())
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
        util.save_network(self.netE, 'E', epoch, self.opt)

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netE = networks.define_E(opt)
        netD = networks.define_D(opt) if opt.isTrain else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            netE = util.load_network(netE, 'E', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)

        return netG, netE, netD
    def preprocess_input(self, data, flag):
        if self.use_gpu():
            P1 = data['P1'].float().cuda()
            if flag == 0 : # gen(P1, B1) -> P1
                B2 = data['B1'].float().cuda()
                P2 = data['P1'].float().cuda()
            elif flag == 1 : # gen(P1, B2) -> P2
                B2 = data['B2'].float().cuda()
                P2 = data['P2'].float().cuda()
            else :
                assert 'flag error'

        return P1, B2, P2
    def feature_matching(self, pred_fake, pred_real):
        num_D = len(pred_fake)
        GAN_Feat_loss = self.FloatTensor(1).fill_(0)
        for i in range(num_D):  # for each discriminator
            # last output is the final prediction, so we exclude it
            num_intermediate_outputs = len(pred_fake[i]) - 1
            for j in range(num_intermediate_outputs):  # for each layer output
                unweighted_loss = self.L1loss(
                    pred_fake[i][j], pred_real[i][j].detach())
                GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
        return GAN_Feat_loss
    def compute_generator_loss(self, image1, bone2, image2):
        self.netG.train()
        self.netE.train()
        self.netD.eval()
        G_losses = defaultdict(int)

        mu, logvar, kld_loss = self.encoding(image1)
        G_losses['KLD_loss'] = kld_loss

        texture_param = [mu, logvar]

        fake_image, l1_loss, vgg_loss, vis = self.generate_fake(texture_param, bone2, image2, cal_loss = True)

        G_losses['L1_loss'] += l1_loss
        G_losses['VGG_loss'] += vgg_loss

        pred_fake, pred_real = self.discriminate(fake_image, image2)

        G_losses['GAN'] = self.GANloss(pred_fake, True, for_discriminator=False)

        if not self.opt.no_ganFeat_loss:
            f_matching = self.feature_matching(pred_fake, pred_real)
            G_losses['GAN_Feat'] = f_matching

        return G_losses, vis

    def discriminate(self, fake, real):
        fake_and_real = torch.cat([fake, real], dim=0)

        discriminator_out = self.netD(fake_and_real)
        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real
    def compute_discriminator_loss(self, image1, bone2, image2):
        self.netG.eval()
        self.netE.eval()
        self.netD.train()
        D_losses = {}
        with torch.no_grad():
            mu, logvar, _ = self.encoding(image1, loss=False)
            texture_param = [mu, logvar]

            fake_image, _, _, _ = self.generate_fake(texture_param, bone2, image2)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        pred_fake, pred_real = self.discriminate(fake_image, image2)

        D_losses['D_fake'] = self.GANloss(pred_fake, False, for_discriminator=True)
        D_losses['D_real'] = self.GANloss(pred_real, True, for_discriminator=True)

        return D_losses
    def encoding(self, image, loss = True):
        mu, logvar = self.netE(image)
        kld_loss = None
        if loss :
            kld_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld

        return mu, logvar, kld_loss
    def generate_fake(self, texture_param, pose_information, image, cal_loss = False):

        l1_loss = 0
        vgg_loss = 0

        gt_vis = []
        fake_vis = []
        vis = None
        x = texture_param
        for step in range(1, self.opt.step_size+1) :
            x, fake_image = self.netG(x, pose_information, texture_param, step)
            if cal_loss :
                ground_truth = self.get_groundtruth(image, step)
                gt_vis.append(ground_truth)
                l1_loss += self.L1loss(fake_image, ground_truth) * self.opt.lambda_rec
                vgg_loss += self.Vggloss(fake_image, ground_truth) * self.opt.lambda_vgg

            fake_vis.append(fake_image.detach())

        if cal_loss :
            gt_vis = torch.cat(gt_vis, -1)
            fake_vis = torch.cat(fake_vis, -1)
            vis = torch.cat([gt_vis, fake_vis], -2)
        return fake_image, l1_loss, vgg_loss, vis
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

    def get_groundtruth(self, img_tensor, step):
        total_step = self.opt.step_size
        if step == total_step : return img_tensor
        dstep = 255 // total_step
        destory_term = 255 - dstep * step

        img_tensor_denorm = (img_tensor + 1) / 2 * 255

        ground_truth = img_tensor_denorm // destory_term * destory_term

        ground_truth = (ground_truth / 255 * 2) - 1

        return ground_truth