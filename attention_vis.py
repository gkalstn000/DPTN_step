import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import util.util as util
from metrics.networks import preprocess_path_for_deform_task
import matplotlib.pyplot as plt
import cv2

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

attention_root = './attention_map'
image_root = '/home/red/external/msha/datasets/fashion'
result_root = './results'
save_path = 'results/attention_vis'

attention_folders = ['DPTN_higher', 'DPTN_higher_spade', 'DPTN_higher_spade_109_can']

gt_dict, distorated_dict = preprocess_path_for_deform_task(os.path.join(image_root, 'test_higher'), os.path.join(result_root, 'DPTN_higher'))

for num in distorated_dict.keys() :
    fake_path_list = distorated_dict[num]
    for fake_img_path in tqdm(fake_path_list, desc = f'num keypoint: {num}') :
        fake_img_name = fake_img_path.split('/')[-1]
        src_img_name, tgt_img_name = fake_img_name.replace('_vis', '').split('_2_')
        src_img_name = src_img_name+'.jpg'

        src_img = Image.open(os.path.join(image_root, 'test_higher', src_img_name)).resize((176, 256))
        tgt_img = Image.open(os.path.join(image_root, 'test_higher', tgt_img_name)).resize((176, 256))
        can_img = Image.open(os.path.join(image_root, 'test_higher_fix_canonical', fake_img_name)).resize((176, 256))
        ex0_img = Image.open(os.path.join(result_root, attention_folders[0], fake_img_name)).resize((176, 256))
        ex1_img = Image.open(os.path.join(result_root, attention_folders[1], fake_img_name)).resize((176, 256))
        ex2_img = Image.open(os.path.join(result_root, attention_folders[2], fake_img_name)).resize((176, 256))
        grid = get_concat_h([tgt_img, ex0_img, ex1_img, ex2_img])

        for i in range(18):
            bonemap = Image.open(
                os.path.join(attention_root, 'DPTN_higher', f'{fake_img_name.replace(".jpg", "")}_{i}_query.jpg')).resize((176, 256))
            bonemap = overlay(tgt_img, bonemap)

            ex0_map = Image.open(os.path.join(attention_root, attention_folders[0],
                                              f'{fake_img_name.replace(".jpg", "")}_{i}_weight.jpg')).resize((176, 256))
            ex0_map = overlay(src_img, ex0_map)
            ex1_map = Image.open(os.path.join(attention_root, attention_folders[1],
                                              f'{fake_img_name.replace(".jpg", "")}_{i}_weight.jpg')).resize((176, 256))
            ex1_map = overlay(src_img, ex1_map)
            ex2_map = Image.open(os.path.join(attention_root, attention_folders[2],
                                              f'{fake_img_name.replace(".jpg", "")}_{i}_weight.jpg')).resize((176, 256))
            ex2_map = overlay(can_img, ex2_map)
            map_grid = get_concat_h([bonemap, ex0_map, ex1_map, ex2_map])
            grid = get_concat_v(grid, map_grid)

        util.mkdir(save_path)
        grid.save(os.path.join(save_path, f'{num}_{fake_img_name}'))
