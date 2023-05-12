"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch.utils.data as data
import torchvision.transforms as transforms
import torch

import numpy as np
from PIL import Image

import random
import json
import util.util as util
import os

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        pass

    def obtain_bone(self, name):
        y, x = self.annotation_file.loc[name]
        coord = util.make_coord_array(y, x)
        return self.obtain_bone_with_coord(coord)
    def obtain_bone_with_coord(self, coord):
        # Keypoint map
        keypoint = util.cords_to_map(coord, self.opt)
        keypoint = torch.Tensor(keypoint)
        # Limb map
        limb = util.limbs_to_map(coord, self.opt)
        limb = torch.Tensor(limb)

        return torch.cat([keypoint, limb])



def __resize(img, w, h, method=Image.BICUBIC):
    return img.resize((w, h), method)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __scale_shortside(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    ss, ls = min(ow, oh), max(ow, oh)  # shortside and longside
    width_is_shorter = ow == ss
    if (ss == target_width):
        return img
    ls = int(target_width * ls / ss)
    nw, nh = (ss, ls) if width_is_shorter else (ls, ss)
    return img.resize((nw, nh), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    return img.crop((x1, y1, x1 + tw, y1 + th))


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def print_gray(path, moduler) :
    img_gray = Image.open(path).convert('L')
    img_array = np.array(img_gray)
    img_clip = img_array // moduler * moduler
    util.print_PILimg(Image.fromarray(img_clip))