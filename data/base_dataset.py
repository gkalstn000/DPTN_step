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
    def obtain_bone_pos(self, name):
        y, x = self.annotation_file.loc[name]
        coord = util.make_coord_array(y, x)

        relative_pos_matrix = []
        for h, w in coord :
            if (h <= -1 or w <= -1) or (not (0<= h < self.opt.old_size[0] and 0<=w<self.opt.old_size[1])) :
                relative_pos_matrix.append(torch.zeros(1, self.opt.load_size, self.opt.load_size))
                continue
            h = int(h / self.opt.old_size[0] * self.opt.load_size)
            w = int(w / self.opt.old_size[1] * self.opt.load_size)

            h_index = self.opt.load_size - h
            w_index = self.opt.load_size - w
            matrix = self.Positional_matrix[:, h_index: h_index + self.opt.load_size, w_index: w_index + self.opt.load_size]
            assert matrix.shape == (1, self.opt.load_size, self.opt.load_size), print(f'({h_index}, {w_index}) / ({h, w})')
            relative_pos_matrix.append(matrix)
        relative_pos_matrix = torch.concatenate(relative_pos_matrix)
        return relative_pos_matrix
    def obtain_bone(self, name):
        if '_2_' in name :
            y, x = self.annotation_file_canonical.loc[name]
        else :
            y, x = self.annotation_file.loc[name]
        coord = util.make_coord_array(y, x)
        return self.obtain_bone_with_coord(coord)
    def obtain_bone_with_coord(self, coord):
        # Keypoint map
        keypoint = util.cords_to_map(coord, self.opt)
        keypoint = np.transpose(keypoint, (2, 0, 1))
        keypoint = torch.Tensor(keypoint)
        if self.opt.pose_nc == 18 :
            return keypoint
        # Limb map
        limb = util.limbs_to_map(coord, self.opt)
        limb = np.transpose(limb, (2, 0, 1))
        limb = torch.Tensor(limb)

        return torch.cat([keypoint, limb])
    def get_canonical_pose(self):
        return ['[28, 54, 54, 93, 130, 55, 95, 131, 117, 180, 233, 117, 178, 230, 24, 23, 27, 26]',
                '[88, 88, 67, 66, 63, 108, 111, 119, 78, 82, 81, 103, 100, 91, 84, 93, 77, 100]']


    def coord_to_PIL(self, coord) :
        heatmap = self.obtain_bone_with_coord(coord)
        heatmap = util.map_to_img(heatmap)
        heatmap_array = (heatmap.numpy() * 255).astype(np.uint8)
        return Image.fromarray(heatmap_array)
def get_transform(B, T):
    B_S_transformed = np.matmul(np.hstack([B, np.ones((B.shape[0], 1))]), T.T)
    B_S_transformed = B_S_transformed[:, :2] / B_S_transformed[:, 2:]
    return B_S_transformed
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