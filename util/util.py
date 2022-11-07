"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import numpy as np
from PIL import Image

import os
import json


LIMB_SEQ = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9],
           [9,10], [1,11], [11,12], [12,13], [1,0], [0,14], [14,16],
           [0,15], [15,17], [0, 16], [0, 17]]

def make_coord_array(keypoint_y, keypoint_x):
    # [[x1, y1], [x2, y2], ..., [x18, y18]] 형식으로 만들기
    keypoint_y = json.loads(keypoint_y)
    keypoint_x = json.loads(keypoint_x)
    keypoint_x = [x+40 if x != -1 else x for x in keypoint_x]
    return np.concatenate([np.expand_dims(keypoint_y, -1), np.expand_dims(keypoint_x, -1)], axis=1)



# ============================ for heatmap ============================
# =====================================================================
def cords_to_map(cords, img_size, sigma=6):
    '''
    :param cords: keypoint coordinates / type: np.array/ shape: (B, 18, 2)
    :param img_size: load image size/ type: tuple/ (H, W)
    :param sigma: scale of heatmap, large sigma makes bigger heatmap
    :return: keypoint(joint) heatmaps/ type: np.array/ shape: (B, H, W, 18)
    '''
    cords = cords.astype(float)
    result = np.zeros(img_size + cords.shape[0:1], dtype='float32')
    for i, point in enumerate(cords):
        if point[0] == -1 or point[1] == -1:
            continue
        point_0 = int(point[0])
        point_1 = int(point[1])
        xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
        result[..., i] = np.exp(-((yy - point_0) ** 2 + (xx - point_1) ** 2) / (2 * sigma ** 2))
    return result

def limbs_to_map(cords, img_size, sigma=8) :
    '''
    :param cords: keypoint coordinates / type: np.array/ shape: (B, 18, 2)
    :param img_size: load image size/ type: tuple/ (H, W)
    :param sigma: scale of heatmap, large sigma makes bigger heatmap
    :return: limb line heatmaps/ type: np.array/ shape: (B, H, W, 19)
    '''
    cords = cords.astype(float)
    result = np.zeros(list(img_size) + [len(LIMB_SEQ)], dtype='float32')
    for i, (src, tgt) in enumerate(LIMB_SEQ) :
        src_point = cords[src]
        tgt_point = cords[tgt]
        if src_point[0] == -1 or src_point[1] == -1 or tgt_point[0] == -1 or tgt_point[1] == -1:
            continue
        trajectories = Bresenham_line(src_point, tgt_point)
        tmp_tensor = np.zeros(list(img_size) + [len(trajectories)], dtype='float32')
        for j, point in enumerate(trajectories) :
            point_0 = int(point[0])
            point_1 = int(point[1])
            xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
            tmp_tensor[..., j] = np.exp(-((yy - point_0) ** 2 + (xx - point_1) ** 2) / (2 * sigma ** 2))
        result[..., i] = tmp_tensor.max(-1)
    return result

def Bresenham_line(p0, p1):
    "Bresenham's line algorithm"
    x0, y0 = p0
    x1, y1 = p1
    points_in_line = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points_in_line.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points_in_line.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points_in_line.append((x, y))
    return points_in_line
# ============================ for heatmap ============================
# =====================================================================

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_image_from_array(array, filename) :
    img = Image.fromarray((array.numpy().max(0) * 255).astype(np.uint8))
    mkdirs('./tmp')
    img.save(f'tmp/{filename}.jpg')

# model parameter I/O
def load_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_dir = os.path.join(opt.checkpoints_dir, opt.id)
    save_path = os.path.join(save_dir, save_filename)
    weights = torch.load(save_path)
    net.load_state_dict(weights)
    return net
def save_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_path = os.path.join(opt.checkpoints_dir, opt.id, save_filename)
    torch.save(net.cpu().state_dict(), save_path)
    if len(opt.gpu_ids) and torch.cuda.is_available():
        net.cuda()
# model parameter I/O
