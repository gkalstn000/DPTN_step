"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import ntpath
import time

import numpy as np

from . import util
from . import html
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
import torch
import wandb

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x

class Visualizer():
    def __init__(self, opt):
        wandb.login()
        wandb.init(project="DPTN", name=opt.id, settings=wandb.Settings(code_dir="."), resume=False)
        self.wandb = wandb
        self.opt = opt
        self.tf_log = opt.isTrain and opt.tf_log
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.id = opt.id
        if self.tf_log:
            self.log_dir = os.path.join(opt.checkpoints_dir, opt.id, 'logs')
            self.writer = SummaryWriter(self.log_dir)


        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.id, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        if opt.isTrain:
            self.log_name = os.path.join(opt.checkpoints_dir, opt.id, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, step):

        ## convert tensors to numpy arrays
        visuals = self.convert_visuals_to_numpy(visuals)

        if self.tf_log:  # show images in tensorboard output
            img_summaries = []
            for label, image_numpy in visuals.items():
                # Create an Image object
                image = Image.fromarray(image_numpy.astype(np.uint8))
                self.wandb.log({label: self.wandb.Image(image)})

                # self.writer.add_image(label, torch.tensor(image_numpy.transpose((2, 0, 1))), step)
                # self.writer.flush()


        if self.use_html:  # save images to a html file
            for label, image_numpy in visuals.items():
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = os.path.join(self.img_dir, 'epoch%.3d_iter%.3d_%s_%d.png' % (epoch, step, label, i))
                        util.save_image(image_numpy[i], img_path)
                else:
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_iter%.3d_%s.png' % (epoch, step, label))
                    if len(image_numpy.shape) >= 4:
                        image_numpy = image_numpy[0]
                    util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment id = %s' % self.id, refresh=5)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    if isinstance(image_numpy, list):
                        for i in range(len(image_numpy)):
                            img_path = 'epoch%.3d_iter%.3d_%s_%d.png' % (n, step, label, i)
                            ims.append(img_path)
                            txts.append(label + str(i))
                            links.append(img_path)
                    else:
                        img_path = 'epoch%.3d_iter%.3d_%s.png' % (n, step, label)
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)
                if len(ims) < 10:
                    webpage.add_images(ims, txts, links, width=self.win_size)
                else:
                    num = int(round(len(ims) / 2.0))
                    webpage.add_images(ims[:num], txts[:num], links[:num], width=self.win_size)
                    webpage.add_images(ims[num:], txts[num:], links[num:], width=self.win_size)
            webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        if self.tf_log:
            self.wandb.log(errors)
            # for tag, value in errors.items():
            #     value = value.mean().float()
            #
            #
            #     self.writer.add_scalar(tag, value.detach().cpu(), step)
            #     self.writer.flush()


    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            # print(v)
            # if v != 0:
            v = v.mean().float()
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def convert_visuals_to_numpy(self, visuals, display_num = 3):
        # image shape : (B, C, H, W)
        for key, t in visuals.items():
            t = t[:display_num]
            tile = self.opt.batchSize > 1
            if 'bone' in key:
                t = util.tensor2label(t, tile=tile)
            else:
                t = util.tensor2im(t, tile=tile)
            visuals[key] = t
        return visuals

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        visuals = self.convert_visuals_to_numpy(visuals)

        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = os.path.join(label, '%s.png' % (name))
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path, create_dir=True)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)

