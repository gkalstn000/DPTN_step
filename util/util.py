"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import numpy as np
from PIL import Image
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
import os
import json
import cv2
import math

LIMB_SEQ = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9],
           [9,10], [1,11], [11,12], [12,13], [1,0], [0,14], [14,16],
           [0,15], [15,17], [0, 16], [0, 17]]
def get_concat_h(imgs):
    width, height = imgs[0].size
    dst = Image.new('RGB', (width * len(imgs), height))
    for i, img in enumerate(imgs) :
        dst.paste(img, (i*width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def overlay(img, weight) :
    added_image = cv2.addWeighted(np.array(img),0.3,np.array(weight.convert('RGB')),1,0)
    return Image.fromarray(added_image)
def make_coord_array(keypoint_y, keypoint_x):
    # [[x1, y1], [x2, y2], ..., [x18, y18]] 형식으로 만들기
    keypoint_y = json.loads(keypoint_y)
    keypoint_x = json.loads(keypoint_x)
    return np.concatenate([np.expand_dims(keypoint_y, -1), np.expand_dims(keypoint_x, -1)], axis=1)

def make_map_videos() :
    map_list = []
    for h in range(32) :
        for w in range(32) :
            map_list.append(point_to_map((w, h), size = (32, 32), sigma=1))
    return np.stack(map_list, axis = 0)
def tensor_to_PIL(tensor) :
    # tensor image [-1, 1]
    image = to_pil_image((tensor.cpu().squeeze() + 1) / 2)
    return image

def src_feature_map_video(tensors, save_path) :
    feature_maps = tensors.squeeze()
    frames = []
    for map in feature_maps:
        min_val = map.min()
        max_val = map.max()
        map = (map - min_val) / (max_val - min_val)
        map_array = (map.cpu() * 255).numpy().astype(np.uint8)
        frames.append(Image.fromarray(map_array))
    frames_to_video(frames, save_path)
def frames_to_video(frames, save_path) :
    videodims = frames[0].size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(save_path, fourcc, 60, videodims)
    # draw stuff that goes on every frame here
    for frame in frames:
        video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    video.release()
def make_attention_video(tgt_img, src_img, attention_weight, maps, save_path) :
    video_frames = []
    tgt_PIL = tensor_to_PIL(tgt_img)
    src_PIL = tensor_to_PIL(src_img)

    for weight, map in zip(attention_weight, maps) :
        weight_img = weight_to_image(weight)
        map_img = Image.fromarray(map.astype(np.uint8)).resize(weight_img.size)
        left = overlay(tgt_PIL, map_img)
        right = overlay(src_PIL, weight_img)
        img_frame = get_concat_h([left, right])
        video_frames.append(img_frame)
    frames_to_video(video_frames, save_path)

# ============================ for heatmap ============================
# =====================================================================
def point_to_map(point, size=[256, 256], sigma=6) :
    x, y = point
    # x = int(x / 176 * 256)

    xx, yy = np.meshgrid(np.arange(size[1]), np.arange(size[0]))
    result = np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma ** 2))

    return Image.fromarray((result * 255).astype(np.uint8))


def cords_to_map(cords, opt, sigma=6):
    '''
    :param cords: keypoint coordinates / type: np.array/ shape: (B, 18, 2)
    :param img_size: load image size/ type: tuple/ (H, W)
    :param sigma: scale of heatmap, large sigma makes bigger heatmap
    :return: keypoint(joint) heatmaps/ type: np.array/ shape: (B, H, W, 18)
    '''
    if isinstance(opt.load_size, int):
        img_size = (opt.load_size, opt.load_size)
    else :
        img_size = opt.load_size
    old_size = opt.old_size
    cords = cords.astype(float)
    result = np.zeros(img_size + cords.shape[0:1], dtype='float32')
    for i, point in enumerate(cords):
        if point[0] <= -1 or point[1] <= -1:continue
        if not (0 <= point[0] < 256 and 0 <= point[1] < 176): continue
        point[0] = point[0]/old_size[0] * img_size[0]
        point[1] = point[1]/old_size[1] * img_size[1]
        point_0 = int(point[0])
        point_1 = int(point[1])
        xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
        result[..., i] = np.exp(-((yy - point_0) ** 2 + (xx - point_1) ** 2) / (2 * sigma ** 2))
    return result

def limbs_to_map(cords, opt, sigma=3) :
    '''
    :param cords: keypoint coordinates / type: np.array/ shape: (B, 18, 2)
    :param img_size: load image size/ type: tuple/ (H, W)
    :param sigma: scale of heatmap, large sigma makes bigger heatmap
    :return: limb line heatmaps/ type: np.array/ shape: (B, H, W, 19)
    '''
    if isinstance(opt.load_size, int):
        img_size = (opt.load_size, opt.load_size)
    else:
        img_size = opt.load_size
    old_size = opt.old_size
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
            point[0] = point[0] / old_size[0] * img_size[0]
            point[1] = point[1] / old_size[1] * img_size[1]
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
            points_in_line.append([x, y])
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points_in_line.append([x, y])
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points_in_line.append([x, y])
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

import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch import nn
# ============= Attention Map vis ==============
def weight_to_image(weight):
    min_value, max_value = weight.min(), weight.max()
    weight = (weight - min_value) / (max_value - min_value)
    weight = weight.view(32, 32) * 255
    weight_image = Image.fromarray(weight.numpy().astype(np.uint8)).convert('L')
    return weight_image.resize((256, 256)) #* color[None, :, None, None]

def save_heatmap(weight, path) :
    fig = plt.figure()
    sns.heatmap(weight.squeeze(), vmin=0.0, vmax=1.0)
    plt.savefig(path)

def bonemap_emphasis(heatmap, index) :
    heatmap[heatmap < 0.5] = 0
    # heatmap[heatmap >= 0.7] = 1
    colors = [[255, 255, 255], [0,255,0]]
    heatmap_color = heatmap.cpu() * torch.Tensor(colors[0])[:, None, None, None]
    heatmap_color[:, index, :, :] = heatmap_color[:, index, :, :] / \
                                    torch.Tensor(colors[0])[:, None,None] * torch.Tensor(colors[1])[:, None, None]
    heatmap_color, _ = heatmap_color.max(1)
    return heatmap_color

# ============= Attention Map vis ==============

def save_image(image_numpy, image_path, create_dir=False):
    if create_dir:
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
    if len(image_numpy.shape) == 2:
        image_numpy = np.expand_dims(image_numpy, axis=2)
    if image_numpy.shape[2] == 1:
        image_numpy = np.repeat(image_numpy, 3, 2)
    if image_numpy.shape[0] == 3 :
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
    image_pil = Image.fromarray(image_numpy)#.resize((176, 256))

    # save image
    image_pil.save(image_path)

def map_to_img(tensor, threshold = 0.5) :
    '''
    :param tensor: (B, C, H, W)
    :param threshold:
    :return:
    '''
    tensor_img, _ = tensor.max(len(tensor.size()) - 3)
    tensor_img[tensor_img < threshold] = 0
    return tensor_img
def tensor2label(tensor, tile) :

    # tensor[tensor < 0.5] = 0
    # color_list = [[240,248,255], [127,255,212], [69,139,116], [227,207,87], [255,228,196], [205,183,158],
    #               [0,0,255], [138,43,226], [255,64,64], [139,35,35], [255,211,155], [138,54,15],
    #               [95,158,160], [122,197,205], [237,145,33], [102,205,0], [205,91,69], [153,50,204]]
    # limb_color = [[174, 58, 231] for _ in range(19)]
    # if tensor.size(1) != 18 :
    #     color_tensor = torch.Tensor(color_list+limb_color)
    # else :
    #     color_tensor = torch.Tensor(color_list)
    # color_tensor = color_tensor.unsqueeze(0)
    # color_tensor = color_tensor.unsqueeze(2)
    # color_tensor = color_tensor.unsqueeze(3)

    tensor = tensor.unsqueeze(4)
    # tensor = (tensor * color_tensor)
    # tensor = tensor.sum(1)
    tensor, _ = tensor.max(1)
    return tensor2im(torch.permute(tensor, (0, 3, 1, 2)), normalize=False, tile=tile)
    # return tensor2im(tensor.to(torch.uint8), tile=tile)



# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True, tile=False):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if image_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(image_tensor.size(0)):
            one_image = image_tensor[b]
            one_image_np = tensor2im(one_image)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tensor = torch.tensor(images_np.transpose((0, 3, 1, 2)))
            images_grid = make_grid(images_tensor, nrow=images_np.shape[0] // 2 + 1)
            return torch.permute(images_grid, (1, 2, 0)).numpy()
        else:
            return images_np[0].transpose((2, 0, 1))

    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)
    image_numpy = image_tensor.detach().cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    # if image_numpy.shape[2] == 1:
    #     image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)

def tile_images(imgs, picturesPerRow=20):
    """ Code borrowed from
    https://stackoverflow.com/questions/26521365/cleanly-tile-numpy-array-of-images-stored-in-a-flattened-1d-format/26521997
    """

    # Padding
    if imgs.shape[0] % picturesPerRow == 0:
        rowPadding = 0
    else:
        rowPadding = picturesPerRow - imgs.shape[0] % picturesPerRow
    if rowPadding > 0:
        imgs = np.concatenate([imgs, np.zeros((rowPadding, *imgs.shape[1:]), dtype=imgs.dtype)], axis=0)

    # Tiling Loop (The conditionals are not necessary anymore)
    tiled = []
    for i in range(0, imgs.shape[0], picturesPerRow):
        tiled.append(np.concatenate([imgs[j] for j in range(i, i + picturesPerRow)], axis=1))

    tiled = np.concatenate(tiled, axis=0)
    return tiled

def print_PILimg(img) :
    plt.imshow(img)
    plt.show()

def positional_encoding(pos, d_model):
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    encoding = torch.zeros(d_model)
    encoding[0::2] = torch.cos(pos * div_term)
    encoding[1::2] = torch.sin(pos * div_term)
    encoding = encoding
    return encoding

def positional_matrix(pos, d_model) :
    encoding_list = torch.stack([positional_encoding(p, d_model) for p in range(pos)])
    matrix = torch.zeros(d_model, pos, pos)
    for i in range(pos) :
        matrix[:, i, i] = encoding_list[i]
        for j in range(i) :
            matrix[:, j, i] = encoding_list[i]
            matrix[:, i, j] = encoding_list[i]

    matrix_expand = torch.zeros(d_model, pos*2, pos*2)
    matrix_expand[:, pos:, pos:] = matrix
    matrix_expand[:, :pos, pos:] = torch.flip(matrix, dims=[1])
    matrix_expand[:, pos:, :pos] = torch.flip(matrix, dims=[2])
    matrix_expand[:, :pos, :pos] = torch.flip(matrix, dims=[1, 2])
    return matrix_expand