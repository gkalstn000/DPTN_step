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


opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = models.create_model(opt)
model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.id,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.id, opt.phase, opt.which_epoch))



# test
for i, data_i in tqdm(enumerate(dataloader)):
    fake_target, fake_source = model(data_i, mode='inference')

    img_path = data_i['path']
    for b in range(fake_target.shape[0]):
        # print('process image... %s' % img_path[b])
        visuals = OrderedDict([('src_image', data_i['src_image'].cpu()),
                                   ('canonical_image', data_i['canonical_image'].cpu()),
                                   ('tgt_map', data_i['tgt_map'].cpu()),
                                   ('real_image', data_i['tgt_image'].cpu()),
                                   ('synthesized_target_image', fake_target.cpu()),
                                   ])
        visualizer.save_images(webpage, visuals, img_path[b:b + 1])
webpage.save()