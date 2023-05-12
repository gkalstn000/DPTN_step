import models
import torch
from models.spade_networks.sync_batchnorm import DataParallelWithCallback



class Trainer() :
    def __init__(self, opt):
        super(Trainer, self).__init__()
        self.opt = opt
        self.model = models.create_model(opt)
        self.model = DataParallelWithCallback(self.model, device_ids = opt.gpu_ids)

        self.generated = None
        if opt.isTrain :
            self.optimizer_G, self.optimizer_D = \
                self.model.module.create_optimizers(opt)
            self.old_lr = opt.lr


    def run_generator_one_step(self, data):
        for i in range(2) :
            self.optimizer_G.zero_grad()
            g_losses, fake_t = self.model(data, mode='generator', flag=i)
            g_loss = sum(g_losses.values()).mean()
            g_loss.backward()
            self.optimizer_G.step()
        self.g_losses = g_losses
        self.generated = fake_t

    def run_discriminator_one_step(self, data):
        for i in range(2) :
            self.optimizer_D.zero_grad()
            d_losses = self.model(data, mode='discriminator', flag=i)
            d_loss = sum(d_losses.values()).mean()
            d_loss.backward()
            self.optimizer_D.step()
        self.d_losses = d_losses

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.generated

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.model.module.save(epoch)

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr

def get_valid_bone_tensors(dataloader_val, model, texture_tensor, bone_tensor) :
    valid_bone_list = ['fashionMENShirts_Polosid0000186605_1front.jpg',
                       'fashionMENSweatshirts_Hoodiesid0000088102_2side.jpg',
                       'fashionMENTees_Tanksid0000595506_2side.jpg',
                       'fashionWOMENBlouses_Shirtsid0000198103_2side.jpg',
                       'fashionWOMENBlouses_Shirtsid0000374902_3back.jpg']
    texture_tensor = torch.unsqueeze(texture_tensor, 0)
    bone_tensor = torch.unsqueeze(bone_tensor, 0)
    valid_bone_tensors = [dataloader_val.dataset.obtain_bone(pose)[None, :, :, :] for pose in valid_bone_list]
    encoder = model.netE
    generator = model.netG

    model.eval()
    output = []
    with torch.no_grad() :
        mu, var = encoder(texture_tensor)
        for bone in [bone_tensor] + valid_bone_tensors :
            output.append(generator([mu, var], bone.cuda()))

    img_grid = torch.cat([texture_tensor] + output, -1)

    bone_grid = [torch.zeros_like(bone_tensor), bone_tensor.cpu()] + valid_bone_tensors
    bone_grid = torch.cat([b.cpu() for b in bone_grid], -1)
    bone_grid, _ = bone_grid.max(1, keepdims=True)
    bone_grid = bone_grid.repeat(1, 3, 1,1)

    results = torch.cat([bone_grid.cpu(), img_grid.cpu()], -2)
    return results


