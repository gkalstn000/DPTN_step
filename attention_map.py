import data
from options.test_options import TestOptions
import models
import numpy as np
import torch
from tqdm import tqdm
from util.visualizer import Visualizer
from util import html
from collections import OrderedDict
import os
import pandas as pd
import util.util as util
import torchvision.transforms as T


transform = T.ToPILImage()
opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = models.create_model(opt)
model.eval()

annotation_file = pd.read_csv(dataloader.dataset.bone_file, sep=':').set_index('name')
annotation_file_canonical = pd.read_csv(dataloader.dataset.bone_file.replace(opt.phase, opt.phase + '-canonical'), sep=':').set_index('name')

# test
for i, data_i in enumerate(tqdm(dataloader)):
    fake_target, fake_source = model(data_i, mode='inference')
    src_name, tgt_name = data_i['path'][0].replace('_vis.jpg', '').split('_2_')
    tgt_y, tgt_x = annotation_file.loc[tgt_name+'.jpg']
    tgt_coord = util.make_coord_array(tgt_y, tgt_x)

    for index, (y, x) in enumerate(tgt_coord) :
        if y == -1 or x == -1 :
            blank = torch.zeros((3, 256, 256))
            transform(blank).save(os.path.join(opt.results_dir, f'{filename}_{index}_query.jpg'))
            transform(blank).save(os.path.join(opt.results_dir, f'{filename}_{index}_weight.jpg'))
        query_index = int(y//8 * 32 + (x * 256 / 176)//8)
        weight = model.last_attn_weights.squeeze()[query_index].cpu()
        weight_image = util.weight_to_image(weight)
        bonemap_color = util.bonemap_emphasis(data_i['tgt_map'], index)

        filename = data_i['path'][0].replace('.jpg', '')
        util.mkdir(opt.results_dir)

        transform(bonemap_color).save(os.path.join(opt.results_dir, f'{filename}_{index}_query.jpg'))
        weight_image.save(os.path.join(opt.results_dir, f'{filename}_{index}_weight.jpg'))
        # util.save_heatmap(upsampling_weight, os.path.join(path, f'{filename}_weight_{index}.jpg'))


    # img_path = data_i['path']
    # for b in range(fake_target.shape[0]):
    #     # print('process image... %s' % img_path[b])
    #     visuals = OrderedDict([('src_image', data_i['src_image'].cpu()),
    #                                ('canonical_image', data_i['canonical_image'].cpu()),
    #                                ('tgt_map', data_i['tgt_map'].cpu()),
    #                                ('real_image', data_i['tgt_image'].cpu()),
    #                                ('synthesized_target_image', fake_target.cpu()),
    #                                ])
    #     visualizer.save_images(webpage, visuals, img_path[b:b + 1])
