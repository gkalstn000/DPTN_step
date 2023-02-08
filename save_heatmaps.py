import os
import sys

import pandas as pd
import numpy as np
from data.base_dataset import BaseDataset
import torch
from util.util import mkdirs
from tqdm import tqdm
import argparse


# if __name__ == '__main__' :
parser = argparse.ArgumentParser(description='scripts to compute all statistics')
parser.add_argument('--mode', help='Path to ground truth data', type=str)
parser.add_argument('--pose_nc', default = '37', help='Path to ground truth data', type=int)
parser.add_argument('--canonical', action='store_true', help='generate canonical heatmaps')
parser.set_defaults(load_size=256)
parser.set_defaults(old_size=(256, 176))
opt = parser.parse_args()

mode = opt.mode
root = './datasets/fashion'
save_path = os.path.join(root, f'{mode}_map' if not opt.canonical else f'{mode}_map_canonical')
mkdirs(save_path)
annotation_file = pd.read_csv(os.path.join(root, f'fasion-annotation-{mode}.csv' if not opt.canonical else f'fasion-annotation-{mode}-canonical.csv'), sep=':')
annotation_file = annotation_file.set_index('name')

dataset = BaseDataset()
dataset.opt = opt
dataset.annotation_file = annotation_file
dataset.annotation_file_canonical = annotation_file
for name  in tqdm(annotation_file.index, desc=f'{mode} heatmap processing') :
    map_tensor = dataset.obtain_bone(name)
    torch.save(map_tensor, os.path.join(save_path, name.replace('jpg', 'pt')))
