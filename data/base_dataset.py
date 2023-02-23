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

    def calculate_transformation_matrix(self, Src_name, Tgt_name):
        """
        Calculates the transformation matrix to align source keypoints to target keypoints
        using Singular Value Decomposition (SVD).
        :param source_keypoints: source keypoints of shape (num_keypoints, 2)
        :param target_keypoints: target keypoints of shape (num_keypoints, 2)
        :return: transformation matrix T and its inverse T_inv
        """
        y, x = self.annotation_file.loc[Src_name]
        B_S = util.make_coord_array(y, x)
        y, x = self.annotation_file.loc[Tgt_name]
        B_T = util.make_coord_array(y, x)
        # Remove occluded keypoints from source and target
        mask_S = np.all(B_S != -1, axis=1)
        mask_T = np.all(B_T != -1, axis=1)
        mask = mask_S & mask_T
        source_keypoints = B_S[mask]
        target_keypoints = B_T[mask]

        # Fill in occluded keypoints with estimated position
        avg_S = np.mean(source_keypoints, axis=0)
        avg_T = np.mean(target_keypoints, axis=0)

        # Calculate transformation matrix
        H = np.dot(source_keypoints.T, target_keypoints)
        U, _, V_t = np.linalg.svd(H)
        R = np.dot(V_t.T, U.T)
        if np.linalg.det(R) < 0:
            V_t[1, :] *= -1
            R = np.dot(V_t.T, U.T)
        t = avg_T.T - np.dot(R, avg_S.T)
        TST = np.eye(3)
        TST[:2, :2] = R
        TST[:2, 2] = t
        TST_inv = np.linalg.inv(TST)
        self.check_trans(B_S, B_T, TST, TST_inv)
        return TST, TST_inv

    def check_trans(self, B_S, B_T, TST, TST_inv) :
        mask_S = np.all(B_S != -1, axis=1)
        mask_T = np.all(B_T != -1, axis=1)

        B_S_recon = get_transform(B_T, TST_inv)
        B_T_recon = get_transform(B_S, TST)

        B_S_recon_mask = B_S.copy()
        B_T_recon_mask = B_T.copy()
        B_S_recon_mask[mask_T] = B_S_recon[mask_T]
        B_T_recon_mask[mask_S] = B_T_recon[mask_S]

        B_S_PIL = self.coord_to_PIL(B_S)
        B_T_PIL = self.coord_to_PIL(B_T)
        # B_S_recon_PIL = self.coord_to_PIL(B_S_recon)
        # B_T_recon_PIL = self.coord_to_PIL(B_T_recon)
        B_S_recon_mask_PIL = self.coord_to_PIL(B_S_recon_mask)
        B_T_recon_mask_PIL = self.coord_to_PIL(B_T_recon_mask)

        util.print_PILimg(B_S_PIL)
        util.print_PILimg(B_T_PIL)
        # util.print_PILimg(B_S_recon_PIL)
        # util.print_PILimg(B_T_recon_PIL)
        util.print_PILimg(B_S_recon_mask_PIL)
        util.print_PILimg(B_T_recon_mask_PIL)



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
