import util.util
from options.test_options import TestOptions
import data
import models
import os
import numpy as np
import torch
import torchvision.transforms as T
from tqdm import tqdm
np.random.seed(1234)


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps.mul(std) + mu

opt = TestOptions().parse()
mode = 'test'
opt.phase = mode
dataloader_val = data.create_dataloader(opt, valid = True)
opt.phase = 'test'


test_images = []
with open(os.path.join(opt.dataroot, f'{mode}.lst'), 'r') as f:
    for lines in f:
        lines = lines.strip()
        if lines.endswith('.jpg'):
            test_images.append(lines)


pose_lists = ['fashionWOMENSkirtsid0000351306_1front.jpg',
              'fashionWOMENSkirtsid0000688501_1front.jpg',
              'fashionWOMENSkirtsid0000155815_2side.jpg',
              'fashionWOMENGraphic_Teesid0000118101_3back.jpg',
              'fashionWOMENSkirtsid0000229601_7additional.jpg',
              'fashionWOMENSkirtsid0000321302_7additional.jpg']




bone_lists = [dataloader_val.dataset.obtain_bone(pose)[None, :, :, :] for pose in pose_lists]

trans = T.ToPILImage()
save_root = 'custom/spain_bone_train'
util.util.mkdirs(save_root)


model = models.create_model(opt)
model.eval()

n_samples = 500
for i, data_i in enumerate(tqdm(dataloader_val)) :
    if i == n_samples - 1 : break
    texture, bone, ground_truth = model.preprocess_input(data_i)
    encoder = model.netE
    generator = model.netG

    with torch.no_grad() :
        mu, var = encoder(texture)

        output = []
        for target_bone in [bone] + bone_lists :
            output.append(generator([mu, var], target_bone.cuda()))

    img_grid = [texture] + output
    img_grid = torch.cat(img_grid, -1)
    img_grid = (img_grid + 1) / 2

    bone_grid = [torch.zeros_like(bone), bone.cpu()] + bone_lists
    bone_grid = torch.cat([b.cpu() for b in bone_grid], -1)
    bone_grid, _ = bone_grid.max(1, keepdims=True)
    bone_grid = bone_grid.repeat(1, 3, 1,1)

    results = torch.cat([bone_grid.cpu(), img_grid.cpu()], -2)
    for k in range(results.size(0)) :
        img_path = data_i['path'][k]
        generated_image = trans(results[k].cpu())
        generated_image.save(os.path.join(save_root, img_path))