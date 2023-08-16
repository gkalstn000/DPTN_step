import data
from options.test_options import TestOptions
import models

import torch.nn as nn
import numpy as np
import torch
import torchvision.transforms as T
from tqdm import tqdm
from util.visualizer import Visualizer
from util import html, util
from collections import OrderedDict
import os
from util.util import load_network, save_network

opt = TestOptions().parse()
# dataloader = data.create_dataloader(opt)

# model = models.create_model(opt)

dim = 128
fc_step = nn.Sequential(nn.Linear(dim * 16 * 16, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, opt.step_size),
                                     )

# prev files
prev_save_filename = './checkpoints/step_dptn'
# prev G
G_filename = 'latest_net_G.pth'
netG_params = torch.load(os.path.join(prev_save_filename, G_filename), map_location=lambda storage, loc: storage)
# prev D
G_filename = 'latest_net_D.pth'
netD_params = torch.load(os.path.join(prev_save_filename, G_filename), map_location=lambda storage, loc: storage)

cur_filename = 'latest_net.pth'
cur_params = torch.load(f'./checkpoints/test/{cur_filename}', map_location=lambda storage, loc: storage)

# G copy
for prevG_key, prevG_val in netG_params.items() :
    if prevG_key in cur_params['netG'].keys() :
        assert prevG_val.size() == cur_params['netG'][prevG_key].size(), '같은이름 하지만 size가 다름'
        cur_params['netG'][prevG_key] = prevG_val
    else :
        print(f'{prevG_key} not in cur G')


for prevD_key, prevD_val in netD_params.items() :
    if prevD_key in cur_params['netD'].keys() :
        assert prevD_val.size() == cur_params['netD'][prevD_key].size(), '같은이름 하지만 size가 다름'
        cur_params['netD'][prevD_key] = prevD_val
    else :
        print(f'{prevD_key} not in cur D')


torch.save(cur_params, f'./checkpoints/step_dptn/pretrained_step_dptn.pth')
print('saved pretrained_step_dptn')
# cur G
# cur D
# cur OptG
# cur optD