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


class MarketDataset(BaseDataset) :

    @staticmethod
    def modify_commandline_options(parser, is_train) :
        parser.set_defaults(load_size=(128, 64))
        parser.set_defaults(old_size=(128, 64))
        parser.set_defaults(image_nc=3)
        parser.set_defaults(display_winsize=128)
        parser.set_defaults(crop_size=128)
        return parser

    def initialize(self, opt):
        super().initialize(opt)
        self.image_dir, self.bone_file, self.name_pairs = self.get_paths(opt)
        size = len(self.name_pairs)
        self.dataset_size = size

        if isinstance(opt.load_size, int):
            self.load_size = (opt.load_size, opt.load_size)
        else:
            self.load_size = opt.load_size
        self.annotation_file = pd.read_csv(self.bone_file, sep=':').set_index('name')

    def get_paths(self, opt):
        root = os.path.join(opt.dataroot, opt.dataset_mode)
        pairLst = os.path.join(root, f'market-pairs-{self.phase}.csv')
        name_pairs = self.init_categories(pairLst)

        image_dir = os.path.join(root, f'{self.phase}')
        bonesLst = os.path.join(root, f'market-annotation-{self.phase}.csv')
        return image_dir, bonesLst, name_pairs

    def postprocess(self, input_dict):
        return input_dict

    def resize_bone(self, bone):
        bone_resized = torch.nn.functional.interpolate(bone[None, :, :, 40:-40], (256, 256))
        return bone_resized.squeeze()

    def __len__(self):
        return self.dataset_size

    def check_bone_img_matching(self, image, bone, save_path = './tmp/img+bone.jpg'):
        import torchvision.transforms as T
        # image : (C, H, W) Tensor
        # bone : (C, H, W) Tensor
        bone_max, _ = bone.max(axis=0, keepdims=True)
        bone_max[bone_max < 0.5] = 0
        Image_Bone = torch.maximum(image, bone_max)

        transform = T.ToPILImage()
        img = transform(Image_Bone)
        img.save(save_path)

