import os
import pandas as pd
import numpy as np
from data.base_dataset import BaseDataset
import torch
from util.util import mkdirs
from tqdm import tqdm

mode = 'train'
root = './datasets/fashion'
save_path = os.path.join(root, f'{mode}_map')
mkdirs(save_path)
annotation_file = pd.read_csv(os.path.join(root, f'fasion-annotation-{mode}.csv'), sep=':')
annotation_file = annotation_file.set_index('name')

dataset = BaseDataset()
dataset.annotation_file = annotation_file
for name  in tqdm(annotation_file.index) :
    map_tensor = dataset.obtain_bone(name, (256, 256))
    torch.save(map_tensor, os.path.join(save_path, name.replace('jpg', 'pt')))
