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


attention_root = './attention_map'
image_root = '/home/red/external/msha/datasets/fashion'
result_root = './results'
save_path = 'results/attention_vis'

attention_folders = ['DPTN_higher',
                     'DPTN_higher_fix_2stage',
                     'DPTN_higher_nonfix_2stage',
                     'DPTN_higher_spade',
                     'DPTN_higher_spade_ex1',
                     'DPTN_higher_spade_ex2',
                     'DPTN_higher_spade_ex3']

gt_dict, distorated_dict = preprocess_path_for_deform_task(os.path.join(image_root, 'test_higher'), os.path.join(result_root, 'DPTN_higher'))

for num in distorated_dict.keys() :
    fake_path_list = distorated_dict[num]
    for fake_img_path in tqdm(fake_path_list, desc = f'num keypoint: {num}') :
        fake_img_name = fake_img_path.split('/')[-1]
        src_img_name, tgt_img_name = fake_img_name.replace('_vis', '').split('_2_')
        src_img_name = src_img_name+'.jpg'

        src_img = Image.open(os.path.join(image_root, 'test_higher', src_img_name)).resize((176, 256))
        tgt_img = Image.open(os.path.join(image_root, 'test_higher', tgt_img_name)).resize((176, 256))
        fix_can_img = Image.open(os.path.join(image_root, 'test_higher_fix_canonical', fake_img_name)).resize((176, 256))
        nonfix_can_img = Image.open(os.path.join(image_root, 'test_higher_nonfix_canonical', fake_img_name)).resize((176, 256))

        ex0_img = Image.open(os.path.join(result_root, attention_folders[0], fake_img_name)).resize((176, 256))
        ex1_img = Image.open(os.path.join(result_root, attention_folders[1], fake_img_name)).resize((176, 256))
        ex2_img = Image.open(os.path.join(result_root, attention_folders[2], fake_img_name)).resize((176, 256))
        ex3_img = Image.open(os.path.join(result_root, attention_folders[3], fake_img_name)).resize((176, 256))
        ex4_img = Image.open(os.path.join(result_root, attention_folders[4], fake_img_name)).resize((176, 256))
        ex5_img = Image.open(os.path.join(result_root, attention_folders[5], fake_img_name)).resize((176, 256))
        ex6_img = Image.open(os.path.join(result_root, attention_folders[6], fake_img_name)).resize((176, 256))
        grid = util.get_concat_h([tgt_img, ex0_img, ex1_img, ex2_img, ex3_img, ex4_img, ex5_img, ex6_img])

        for i in range(18):
            bonemap = Image.open(os.path.join(attention_root, attention_folders[0],
                                              f'{fake_img_name.replace(".jpg", "")}_{i}_query.jpg')).resize((176, 256))
            bonemap = util.overlay(tgt_img, bonemap)

            ex0_map = Image.open(os.path.join(attention_root, attention_folders[0],
                                              f'{fake_img_name.replace(".jpg", "")}_{i}_weight.jpg')).resize((176, 256))
            ex0_map = util.overlay(src_img, ex0_map)

            ex1_map = Image.open(os.path.join(attention_root, attention_folders[1],
                                              f'{fake_img_name.replace(".jpg", "")}_{i}_weight.jpg')).resize((176, 256))
            ex1_map = util.overlay(fix_can_img, ex1_map)
            try :
                ex2_map = Image.open(os.path.join(attention_root, attention_folders[2],
                                                  f'{fake_img_name.replace(".jpg", "")}_{i}_weight.jpg')).resize((176, 256))
                ex2_map = util.overlay(nonfix_can_img, ex2_map)
            except :
                continue
            ex3_map = Image.open(os.path.join(attention_root, attention_folders[3],
                                              f'{fake_img_name.replace(".jpg", "")}_{i}_weight.jpg')).resize((176, 256))
            ex3_map = util.overlay(src_img, ex3_map)

            ex4_map = Image.open(os.path.join(attention_root, attention_folders[4],
                                              f'{fake_img_name.replace(".jpg", "")}_{i}_weight.jpg')).resize((176, 256))
            ex4_map = util.overlay(fix_can_img, ex4_map)

            ex5_map = Image.open(os.path.join(attention_root, attention_folders[5],
                                              f'{fake_img_name.replace(".jpg", "")}_{i}_weight.jpg')).resize((176, 256))
            ex5_map = util.overlay(src_img, ex5_map)

            ex6_map = Image.open(os.path.join(attention_root, attention_folders[6],
                                              f'{fake_img_name.replace(".jpg", "")}_{i}_weight.jpg')).resize((176, 256))
            ex6_map = util.overlay(src_img, ex6_map)

            map_grid = util.get_concat_h([bonemap, ex0_map, ex1_map, ex2_map, ex3_map, ex4_map, ex5_map, ex6_map])
            grid = util.get_concat_v(grid, map_grid)

        util.mkdir(save_path)
        grid.save(os.path.join(save_path, f'{num}_{fake_img_name}'))
