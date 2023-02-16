import data
from options.test_options import TestOptions
import models
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
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
            y = np.random.randint(64, 196)
            x = np.random.randint(44, 132)
            zero = Image.fromarray(np.zeros((256, 256)))
            zero.convert('L').save(os.path.join(opt.results_dir, f'{filename}_{index}_query.jpg'))
            zero.convert('L').save(os.path.join(opt.results_dir, f'{filename}_{index}_weight.jpg'))
            continue

        query_index = int(y//8 * 32 + x //5.5)
        assert query_index < 1024, 'Query index calculating is wrong'

        weight = model.last_attn_weights.squeeze()[query_index].cpu()
        weight_image = util.weight_to_image(weight)
        bonemap = util.point_to_map((x, y))

        filename = data_i['path'][0].replace('.jpg', '')
        util.mkdir(opt.results_dir)

        bonemap.convert('L').save(os.path.join(opt.results_dir, f'{filename}_{index}_query.jpg'))
        weight_image.convert('L').save(os.path.join(opt.results_dir, f'{filename}_{index}_weight.jpg'))

