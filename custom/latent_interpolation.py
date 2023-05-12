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
mode = 'train'
opt.phase = mode
dataloader_val = data.create_dataloader(opt, valid = True)
opt.phase = 'test'


test_images = []
with open(os.path.join(opt.dataroot, f'{mode}.lst'), 'r') as f:
    for lines in f:
        lines = lines.strip()
        if lines.endswith('.jpg'):
            test_images.append(lines)


n_samples = 500
interval = 10
alphas = torch.linspace(0, 1, interval + 2)
trans = T.ToPILImage()
save_root = 'custom/spain_interpolation_train'
util.util.mkdirs(save_root)

index_pairs = np.unique(np.random.randint(0, len(test_images), size=(n_samples, 2)), axis=0)
name_pairs = [[test_images[from_], test_images[to_]] for from_, to_ in index_pairs]
dataloader_val.dataset.name_pairs = name_pairs

model = models.create_model(opt)
model.eval()
for i, data_i in enumerate(tqdm(dataloader_val)) :
    if i == n_samples - 1 : break
    texture, bone, ground_truth = model.preprocess_input(data_i)
    encoder = model.netE
    generator = model.netG

    with torch.no_grad() :
        mu_src, var_src = encoder(texture)
        mu_tgt, var_tgt = encoder(ground_truth)

        mu_interpolations = [alpha * mu_tgt + (1 - alpha) * mu_src for alpha in alphas]
        var_interpolations = [alpha * var_tgt + (1 - alpha) * var_src for alpha in alphas]
        output = []
        for mu, var in zip(mu_interpolations, var_interpolations) :
            output.append(generator([mu, var], bone))
    output = [texture] + output + [ground_truth]
    interpolation_result = torch.cat(output, -1)
    interpolation_result = (interpolation_result + 1) / 2
    for k in range(interpolation_result.size(0)) :
        img_path = data_i['path'][k]
        generated_image = trans(interpolation_result[k].cpu())
        generated_image.save(os.path.join(save_root, img_path))