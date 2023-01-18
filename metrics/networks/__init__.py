import os
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict
from util import util
from tqdm import tqdm
import glob
import json
from skimage.draw import disk, polygon
import torch.utils.data


def pad_256(img):
    if img.size != (176, 256) :
        img = img.resize((176, 256), Image.BICUBIC)
    img = np.array(img)
    result = np.ones((256, 256, 3), dtype=float) * 255
    result[:,40:216,:] = img
    return result

def make_dataloader(dataloader, batchsize) :
    return  torch.utils.data.DataLoader(
            dataloader,
            batch_size=batchsize,
            num_workers=6,
            shuffle=False
        )

def get_image_list(flist):
    if isinstance(flist, list):
        return flist

    # flist: image file path, image directory path, text file flist path
    if isinstance(flist, str):
        if os.path.isdir(flist):
            flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
            flist.sort()
            return flist

        if os.path.isfile(flist):
            try:
                return np.genfromtxt(flist, dtype=np.str)
            except:
                return [flist]
    print('can not read files from %s return empty list'%flist)
    return []


def compare_l1(img_true, img_test):
    img_true = img_true.astype(np.float32)
    img_test = img_test.astype(np.float32)
    return np.mean(np.abs(img_true - img_test))


def compare_mae(img_true, img_test):
    img_true = img_true.astype(np.float32)
    img_test = img_test.astype(np.float32)
    return np.sum(np.abs(img_true - img_test)) / np.sum(img_true + img_test)

def get_keypoint_dict() :
    keypoint_path = 'datasets/fashion/fasion-annotation-test.csv'
    df_keypoint = pd.read_csv(keypoint_path, sep=':')
    keypoint_dict = defaultdict(list)
    for idx, (name, keypoints_y, keypoints_x) in df_keypoint.iterrows() :
        coord = util.make_coord_array(keypoints_y, keypoints_x)
        num_keypoint = 18 - sum(coord.sum(-1) == -2)
        keypoint_dict[num_keypoint].append(name)
    return keypoint_dict
def check_keypoint(keypoint_dict, img_name) :
    for num_keypoint, img_list in keypoint_dict.items() :
        if img_name in img_list :
            return num_keypoint
    assert 'something wrong with check keypoint'
def preprocess_path_for_deform_task(gt_path, distorted_path):
    distorted_image_list = sorted(get_image_list(distorted_path))
    gt_list= defaultdict(list)
    distorated_list= defaultdict(list)
    keypoint_dict = get_keypoint_dict()

    for distorted_image in tqdm(distorted_image_list, desc='classfing image by keypoints'):
        image = os.path.basename(distorted_image)
        image = image.split('_2_')[-1]
        image = image.split('_vis')[0] + '.jpg' # if '.jpg' not in image else image.split('_vis')[0]
        gt_image = os.path.join(gt_path, image)
        if not os.path.isfile(gt_image):
            print(gt_image)
            continue
        num_keypoint = check_keypoint(keypoint_dict, image.replace('.png', '.jpg'))
        gt_list[num_keypoint].append(gt_image)
        gt_list[19].append(gt_image)
        distorated_list[num_keypoint].append(distorted_image)
        distorated_list[19].append(distorted_image)

    return dict(sorted(gt_list.items())), dict(sorted(distorated_list.items()))


def produce_ma_mask(kp_array, img_size=(128, 64), point_radius=4):
    MISSING_VALUE = -1
    from skimage.morphology import dilation, erosion, square
    mask = np.zeros(shape=img_size, dtype=bool)
    limbs = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10],
              [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17],
               [1,16], [16,18], [2,17], [2,18], [9,12], [12,6], [9,3], [17,18]]
    limbs = np.array(limbs) - 1
    for f, t in limbs:
        from_missing = kp_array[f][0] == MISSING_VALUE or kp_array[f][1] == MISSING_VALUE
        to_missing = kp_array[t][0] == MISSING_VALUE or kp_array[t][1] == MISSING_VALUE
        if from_missing or to_missing:
            continue

        norm_vec = kp_array[f] - kp_array[t]
        norm_vec = np.array([-norm_vec[1], norm_vec[0]])
        norm_vec = point_radius * norm_vec / np.linalg.norm(norm_vec)


        vetexes = np.array([
            kp_array[f] + norm_vec,
            kp_array[f] - norm_vec,
            kp_array[t] - norm_vec,
            kp_array[t] + norm_vec
        ])
        yy, xx = polygon(vetexes[:, 0], vetexes[:, 1], shape=img_size)
        mask[yy, xx] = True

    for i, joint in enumerate(kp_array):
        if kp_array[i][0] == MISSING_VALUE or kp_array[i][1] == MISSING_VALUE:
            continue
        yy, xx = disk((joint[0], joint[1]), radius=point_radius, shape=img_size)
        mask[yy, xx] = True

    mask = dilation(mask, square(5))
    mask = erosion(mask, square(5))
    return mask


def load_pose_cords_from_strings(y_str, x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    return np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)


def create_masked_image(ano_to):
    kp_to = load_pose_cords_from_strings(ano_to['keypoints_y'], ano_to['keypoints_x'])
    mask = produce_ma_mask(kp_to)
    return mask
