from data.base_dataset import BaseDataset
from PIL import Image
import util.util as util
import os
import pandas as pd
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import torch

from tqdm import tqdm, trange
import numpy as np
import time


class FashionDataset(BaseDataset) :

    @staticmethod
    def modify_commandline_options(parser, is_train) :
        parser.set_defaults(load_size=(256, 256))
        parser.set_defaults(old_size=(256, 176))
        parser.set_defaults(image_nc=3)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(crop_size=256)
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.phase = opt.phase
        self.image_dir, self.bone_file, self.name_pairs = self.get_paths(opt)
        size = len(self.name_pairs)
        self.dataset_size = size

        if isinstance(opt.load_size, int):
            self.load_size = (opt.load_size, opt.load_size)
        else:
            self.load_size = opt.load_size


        # transform_list.append(transforms.Resize(size=self.load_size))
        self.annotation_file = pd.read_csv(self.bone_file, sep=':').set_index('name')

        transform_list=[]
        # transform_list.append(transforms.Resize(size=self.load_size))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)))
        self.trans = transforms.Compose(transform_list)

    def get_paths(self, opt):
        root = opt.dataroot
        pairLst = os.path.join(root, f'fashion-pairs-{self.phase}.csv')
        name_pairs = self.init_categories(pairLst)

        image_dir = os.path.join(root, f'{self.phase}_higher')
        bonesLst = os.path.join(root, f'fashion-annotation-{self.phase}.csv')
        return image_dir, bonesLst, name_pairs

    def init_categories(self, pairLst):
        pairs_file_train = pd.read_csv(pairLst)
        size = len(pairs_file_train)
        pairs = []
        for i in trange(size, desc = 'Loading data pairs ...'):
            pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
            pairs.append(pair)

        print('Loading data pairs finished ...')
        return pairs

    def __getitem__(self, index):
        P1_name, P2_name = self.name_pairs[index]
        PC_name = f'{P1_name.replace(".jpg", "")}_2_{P2_name.replace(".jpg", "")}_vis.jpg'

        P1_path = os.path.join(self.image_dir, P1_name) # person 1
        P2_path = os.path.join(self.image_dir, P2_name) # person 2

        if self.phase == 'train' :
            P1_img = Image.open(P1_path).convert('RGB')
            P1_img = F.resize(P1_img, self.load_size)
            texture = self.trans(P1_img)

            bone = self.obtain_bone(P1_name)

            ground_truth = texture

        else :
            P1_img = Image.open(P1_path).convert('RGB')
            P1_img = F.resize(P1_img, self.load_size)
            texture = self.trans(P1_img)

            bone = self.obtain_bone(P2_name)

            P2_img = Image.open(P2_path).convert('RGB')
            P2_img = F.resize(P2_img, self.load_size)
            ground_truth = self.trans(P2_img)

        input_dict = {'texture' : texture,
                      'bone': bone,
                      'ground_truth': ground_truth,
                      'path' : PC_name}


        return input_dict


    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size


