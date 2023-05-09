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

        self.netG, self.netE, self.netD = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.GANloss = loss.GANLoss(opt.gan_mode, tensor=self.FloatTensor, opt=self.opt).cuda()
            self.L1loss = torch.nn.L1Loss()
            self.Vggloss = loss.VGGLoss()
            self.L2loss = torch.nn.MSELoss()
            self.KLDLoss = loss.KLDLoss()


    def forward(self, data, mode):
        texture, bone, ground_truth = self.preprocess_input(data)
        if mode == 'generator':
            g_loss, fake_t = self.compute_generator_loss(texture, bone, ground_truth)
            return g_loss, fake_t
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(texture, bone)
            return d_loss
        elif mode == 'inference' :
            self.netG.eval()
            with torch.no_grad():
                fake_image_t, _ = self.generate_fake(texture, bone)
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
    def preprocess_input(self, data):
        if self.use_gpu():
            data['texture'] = data['texture'].float().cuda()
            data['bone'] = data['bone'].float().cuda()
            data['ground_truth'] = data['ground_truth'].float().cuda()

        return data['texture'], data['bone'], data['ground_truth']

    def compute_generator_loss(self, texture, bone, texture2):
        self.netG.train()
        self.netD.train()
        G_losses = {}

        fake_image, kld_loss = self.generate_fake(texture, bone)
        pred_fake, pred_real = self.discriminate(bone, fake_image, texture)

        G_losses['GAN'] = self.GANloss(pred_fake, True, for_discriminator=False)
        G_losses['VGG_loss'] =  self.Vggloss(fake_image, texture) * self.opt.lambda_vgg
        G_losses['KLD_loss'] = kld_loss
        G_losses['L1_loss'] = self.L1loss(fake_image, texture) * self.opt.lambda_rec

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

        # with torch.no_grad() :
        #     z_dict_tgt = self.netG.z_encoder(texture2)
        # G_losses['z_distance'] = self.L2loss(z_dict['noise'], z_dict_tgt['noise']) * 0.1


        return G_losses, fake_image
    def discriminate(self, bone_map, fake, real):
        fake_concat = torch.cat([bone_map, fake], dim=1)
        real_concat = torch.cat([bone_map, real], dim=1)
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)
        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real
    def compute_discriminator_loss(self, texture, bone):
        self.netG.train()
        self.netD.train()
        D_losses = {}
        with torch.no_grad():
            fake_image_t, _ = self.generate_fake(texture, bone)
            fake_image_t = fake_image_t.detach()
            fake_image_t.requires_grad_()

        pred_fake, pred_real = self.discriminate(bone, fake_image_t, texture)

        D_losses['D_fake'] = self.GANloss(pred_fake, False, for_discriminator=True)
        D_losses['D_real'] = self.GANloss(pred_real, True, for_discriminator=True)

        return D_losses

    def generate_fake(self, texture, bone):
        mu, logvar = self.netE(texture)
        fake_image_t = self.netG([mu, logvar], bone)

        kld_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld

        return fake_image_t, kld_loss
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