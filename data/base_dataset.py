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
        if name :
            y, x = self.annotation_file.loc[name]
        else :
            y, x = self.get_canonical_pose()
        coord = util.make_coord_array(y, x)
        # Keypoint map
        keypoint = util.cords_to_map(coord, self.opt)
        keypoint = np.transpose(keypoint, (2, 0, 1))
        keypoint = torch.Tensor(keypoint)
        # Limb map
        limb = util.limbs_to_map(coord, self.opt)
        limb = np.transpose(limb, (2, 0, 1))
        limb = torch.Tensor(limb)

        return torch.cat([keypoint, limb])


    def get_canonical_pose(self):
        return ['[28, 54, 54, 93, 130, 55, 95, 131, 117, 180, 233, 117, 178, 230, 24, 23, 27, 26]',
                '[88, 88, 67, 66, 63, 108, 111, 119, 78, 82, 81, 103, 100, 91, 84, 93, 77, 100]']

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
