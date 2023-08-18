"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import numpy as np
from PIL import Image
from torchvision.utils import make_grid
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
import os
import json
import cv2
import math
import torch.distributed as dist

def init_distributed():

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)
    # synchronizes all the threads to reach this point before moving on
    dist.barrier()
    setup_for_distributed(rank == 0)
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
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

def is_valid_point(point, old_size) :
    return (0 <= point[0] < old_size[0]) and (0 <= point[1] < old_size[1])
def cords_to_map(cords, opt, sigma=6):
    '''
    :param cords: keypoint coordinates / type: np.array/ shape: (B, 18, 2)
    :param img_size: load image size/ type: tuple/ (H, W)
    :param sigma: scale of heatmap, large sigma makes bigger heatmap
    :return: keypoint(joint) heatmaps/ type: np.array/ shape: (B, H, W, 18)
    '''
    factor = 1
    img_size = (opt.load_size[0] // factor, opt.load_size[1] // factor)
    old_size = opt.old_size
    xx, yy = np.meshgrid(np.arange(img_size[1]*2), np.arange(img_size[0]*2))
    heatmap = np.exp(-((yy - (img_size[0] - 1)) ** 2 + (xx - (img_size[1] - 1)) ** 2) / (2 * sigma ** 2))

    cords = cords.astype(float)
    result = []
    for i, point in enumerate(cords):
        if not is_valid_point(point, old_size) :
            result.append(np.zeros(img_size))
            continue
        h_, w_ = img_size - (point / old_size * img_size + 1).astype(np.int)
        result.append(heatmap[h_:h_+img_size[0], w_:w_+img_size[1]])
        assert heatmap[h_:h_+img_size[0], w_:w_+img_size[1]].shape == img_size, f'point: {point}'

    return np.stack(result)

def limbs_to_map(cords, opt, sigma=3) :
    '''
    :param cords: keypoint coordinates / type: np.array/ shape: (B, 18, 2)
    :param img_size: load image size/ type: tuple/ (H, W)
    :param sigma: scale of heatmap, large sigma makes bigger heatmap
    :return: limb line heatmaps/ type: np.array/ shape: (B, H, W, 19)
    '''
    LIMB_SEQ = [[0, 16], [0,14], [0, 15], [0, 17], [16, 17], [0, 1], # Face lines
                [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [11, 12], [12, 13], # Limb lines
                [1, 2], [2, 8], [8, 11], [11, 5], [5, 1], [2, 11], [1, 8], [1, 11], [5, 8]]
    factor = 1
    img_size = (opt.load_size[0] // factor, opt.load_size[1] // factor)
    old_size = opt.old_size
    heatmap_column = np.exp(-(img_size[0]-1 - np.arange(img_size[0]*2 - 1))**2 / (2 * sigma ** 2))
    heatmap_column = heatmap_column.reshape((heatmap_column.shape[0], 1))
    heatmap_line = np.tile(heatmap_column, (1, img_size[0]))
    pad = np.zeros((img_size[0]*2-1, img_size[0] - 1))
    heatmap_line = np.concatenate([pad, heatmap_line], axis = 1)

    cords = cords.astype(float)
    result = []
    for i, (src, tgt) in enumerate(LIMB_SEQ) :
        heatmap = np.zeros(img_size)
        src_point = cords[src]
        tgt_point = cords[tgt]
        if not (is_valid_point(src_point, old_size) and is_valid_point(tgt_point, old_size)):
            result.append(heatmap)
            continue
        h1, w1 = (src_point / old_size * img_size).astype(np.int)
        h2, w2 = (tgt_point / old_size * img_size).astype(np.int)

        # rotate
        center = (img_size[1] - 1, img_size[0] - 1)
        distance = np.sqrt((h2-h1)**2 + (w2-w1)**2)
        angle = -np.degrees(np.arctan2(h2-h1, w2-w1))
        rot = cv2.getRotationMatrix2D(center, angle, distance / img_size[1])
        rotated = cv2.warpAffine(heatmap_line, rot, (0, 0))

        h_, w_ = img_size - (src_point / old_size * img_size + 1).astype(np.int)
        result.append(rotated[h_:h_+img_size[0], w_:w_+img_size[1]])
        assert rotated[h_:h_+img_size[0], w_:w_+img_size[1]].shape == img_size, f'src: {src_point}, tgt: {tgt_point}'

    return np.stack(result)


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
def load_network(epoch, opt):
    save_filename = '%s_net.pth' % (epoch)
    save_dir = os.path.join(opt.checkpoints_dir, opt.id)
    save_path = os.path.join(save_dir, save_filename)
    ckpt = torch.load(save_path, map_location=lambda storage, loc: storage)

    return ckpt
def save_network(G, D, optG, optD, epoch, opt):
    save_filename = '%s_net.pth' % (epoch)
    save_path = os.path.join(opt.checkpoints_dir, opt.id, save_filename)

    torch.save(
        {
            "netG": G.state_dict(),
            "netD": D.state_dict(),
            "optG": optG.state_dict(),
            "optD": optD.state_dict()
        },
        save_path
    )

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
    color_list = [[240,248,255], [127,255,212], [69,139,116], [227,207,87], [255,228,196], [205,183,158],
                  [0,0,255], [138,43,226], [255,64,64], [139,35,35], [255,211,155], [138,54,15],
                  [95,158,160], [122,197,205], [237,145,33], [102,205,0], [205,91,69], [153,50,204]]
    limb_color = [[174, 58, 231] for _ in range(23)]
    if tensor.size(1) != 18 :
        color_tensor = torch.Tensor(color_list+limb_color)
    else :
        color_tensor = torch.Tensor(color_list)
    color_tensor = color_tensor[None, :, None, None, :]

    tensor = tensor.unsqueeze(4)
    tensor = (tensor * color_tensor)

    edge = torch.zeros(256-4, 256-4)
    edge = F.pad(edge, pad=(2, 2, 2, 2), mode='constant', value=255)
    edge = torch.stack([edge] * 3, dim=2).repeat((tensor.size(0), 1, 1, 1, 1))
    tensor = torch.cat([tensor, edge], dim = 1)
    # tensor = tensor.sum(1)
    tensor, _ = tensor.max(1)
    return tensor2im(torch.permute(tensor / 255, (0, 3, 1, 2)) , normalize=False, tile=tile)
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
            one_image_np = tensor2im(one_image, normalize=normalize)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tensor = torch.tensor(images_np.transpose((0, 3, 1, 2)))
            images_grid = make_grid(images_tensor, nrow= 3)
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


def crop_face_from_output(image, face_center, crop_smaller=0):
    r"""Crop out the face region of the image (and resize if necessary to feed
    into generator/discriminator).

    Args:
        image (NxC1xHxW tensor or list of tensors): Image to crop.
        input_label (list) list of the face center.
        crop_smaller (int): Number of pixels to crop slightly smaller region.
    Returns:
        output (NxC1xHxW tensor or list of tensors): Cropped image.
    """
    if type(image) == list:
        return [crop_face_from_output(im, face_center, crop_smaller)
                for im in image]
    output = None
    face_size = image.shape[-2] // 32 * 8
    for i in range(face_center.shape[0]):
        face_position = get_face_bbox_for_output(
            image,
            face_center[i],
            crop_smaller=crop_smaller)
        if face_position is not None:
            ys, ye, xs, xe = face_position
            output_i = torch.nn.functional.interpolate(
                image[i:i + 1, -3:, ys:ye, xs:xe],
                size=(face_size, face_size), mode='bilinear',
                align_corners=True)
        else:
            output_i = torch.zeros(1, 3, face_size, face_size, device=image.device)
        output = torch.cat([output, output_i]) if i != 0 else output_i
    return output

def get_face_bbox_for_output(image, face_center, crop_smaller=0):
    _,_,h,w = image.shape
    if torch.sum(face_center) != -4:
        xs, ys, xe, ye = face_center
        xc, yc = (xs + xe) // 2, (ys + ye) // 2
        ylen = int((xe - xs) * 2.5)

        ylen = xlen = min(w, max(32, ylen))
        yc = max(ylen // 2, min(h - 1 - ylen // 2, yc))
        xc = max(xlen // 2, min(w - 1 - xlen // 2, xc))

        ys, ye = int(yc) - ylen // 2, int(yc) + ylen // 2
        xs, xe = int(xc) - xlen // 2, int(xc) + xlen // 2
        if crop_smaller != 0:  # Crop slightly smaller region inside face.
            ys += crop_smaller
            xs += crop_smaller
            ye -= crop_smaller
            xe -= crop_smaller
        return [ys, ye, xs, xe]
    else:
        return None