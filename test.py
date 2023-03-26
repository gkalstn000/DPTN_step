import data
from options.test_options import TestOptions
import models
import numpy as np
import torch
import torchvision.transforms as T
from tqdm import tqdm
from util.visualizer import Visualizer
from util import html, util
from collections import OrderedDict
import os


opt = TestOptions().parse()
dataloader = data.create_dataloader(opt)

model = models.create_model(opt)
model.eval()

if not opt.simple_test:
    visualizer = Visualizer(opt)
    # create a webpage that summarizes the all results
    web_dir = os.path.join(opt.results_dir, opt.id,
                           '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir,
                        'Experiment = %s, Phase = %s, Epoch = %s' %
                        (opt.id, opt.phase, opt.which_epoch))

trans = T.ToPILImage()
result_path = os.path.join(opt.results_dir, opt.id)
util.mkdirs(result_path)
# test
for i, data_i in enumerate(tqdm(dataloader)):
    fake_target, fake_source = model(data_i, mode='inference')

    img_path = data_i['path']
    if opt.simple_test:
        fake_target = (fake_target + 1) / 2
        for k in range(fake_target.shape[0]) :
            generated_image = trans(fake_target[k].cpu())
            generated_image.save(os.path.join(result_path, img_path[k]))
        continue
    for b in range(fake_target.shape[0]):
        # print('process image... %s' % img_path[b])
        visuals = OrderedDict([('src_image', data_i['src_image'].cpu()),
                                   ('real_image', data_i['tgt_image'].cpu()),
                                   ('synthesized_target_image', fake_target.cpu()),
                                   ])
        visualizer.save_images(webpage, visuals, img_path[b:b + 1])
if not opt.simple_test:
    webpage.save()