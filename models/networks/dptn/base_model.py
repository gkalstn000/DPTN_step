import os
import torch
import sys
from collections import OrderedDict
import torch.nn as nn
from torch.nn import init
from util import util
import numpy as np
import ntpath
import cv2

class BaseModel():
    def name(self):
        return 'BaseModel'
    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).'
              % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        return self.image_paths

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        """Return visualization images"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                value = getattr(self, name)
                if isinstance(value, list):
                    # visual multi-scale ouputs
                    for i in range(len(value)):
                        visual_ret[name + str(i)] = self.convert2im(value[i], name)
                else:
                    visual_ret[name] =self.convert2im(value, name)
        return visual_ret


    def get_current_errors(self):
        """Return training loss"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = getattr(self, 'loss_' + name).item()
        return errors_ret

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, 'E', epoch, self.opt)


    # load models
    def load_networks(self, which_epoch):
        """Load all the networks from the disk"""
        for name in self.model_names:
            if isinstance(name, str):
                filename = '%s_net_%s.pth' % (which_epoch, name)
                path = os.path.join(self.save_dir, filename)
                net = getattr(self, 'net_' + name)
                try:
                    '''
                    new_dict = {}
                    pretrained_dict = torch.load(path)
                    for k, v in pretrained_dict.items():
                        if 'transformer' in k:
                            new_dict[k.replace('transformer', 'PTM')] = v
                        else:
                            new_dict[k] = v

                    net.load_state_dict(new_dict)
                    '''
                    net.load_state_dict(torch.load(path))
                    print('load %s from %s' % (name, filename))
                except FileNotFoundError:
                    print('do not find checkpoint for network %s'%name)
                    continue
                except:
                    pretrained_dict = torch.load(path)
                    model_dict = net.state_dict()
                    try:
                        pretrained_dict_ = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                        if len(pretrained_dict_) == 0:
                            pretrained_dict_ = {k.replace('module.', ''): v for k, v in pretrained_dict.items() if
                                                k.replace('module.', '') in model_dict}
                        if len(pretrained_dict_) == 0:
                            pretrained_dict_ = {('module.' + k): v for k, v in pretrained_dict.items() if
                                                'module.' + k in model_dict}

                        pretrained_dict = pretrained_dict_
                        net.load_state_dict(pretrained_dict)
                        print('Pretrained network %s has excessive layers; Only loading layers that are used' % name)
                    except:
                        print('Pretrained network %s has fewer layers; The following are not initialized:' % name)
                        not_initialized = set()
                        for k, v in pretrained_dict.items():
                            if v.size() == model_dict[k].size():
                                model_dict[k] = v

                        for k, v in model_dict.items():
                            if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                                # not_initialized.add(k)
                                not_initialized.add(k.split('.')[0])
                        print(sorted(not_initialized))
                        net.load_state_dict(model_dict)
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    net.cuda()
                if not self.isTrain:
                    net.eval()

    def update_learning_rate(self, epoch=None):
        """Update learning rate"""
        for scheduler in self.schedulers:
            if epoch == None:
                scheduler.step()
            else:
                scheduler.step(epoch)
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate=%.7f' % lr)

    def get_current_learning_rate(self):
        lr_G = self.optimizers[0].param_groups[0]['lr']
        lr_D = self.optimizers[1].param_groups[0]['lr']
        return lr_G, lr_D
