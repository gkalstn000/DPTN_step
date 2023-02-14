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

class FashionDataset(BaseDataset) :

    @staticmethod
    def modify_commandline_options(parser, is_train) :
        if is_train:
            parser.set_defaults(load_size=256)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(old_size=(256, 176))
        parser.set_defaults(image_nc=3)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(crop_size=256)
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.phase = opt.phase
        self.image_dir, self.canonical_dir, self.bone_file, self.name_pairs = self.get_paths(opt)
        size = len(self.name_pairs)
        self.dataset_size = size

        if isinstance(opt.load_size, int):
            self.load_size = (opt.load_size, opt.load_size)
        else:
            self.load_size = opt.load_size


        # transform_list.append(transforms.Resize(size=self.load_size))
        self.annotation_file = pd.read_csv(self.bone_file, sep=':')
        self.annotation_file = self.annotation_file.set_index('name')

        transform_list=[]
        # transform_list.append(transforms.Resize(size=self.load_size))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)))
        self.trans = transforms.Compose(transform_list)

    def get_paths(self, opt):
        root = opt.dataroot
        pairLst = os.path.join(root, f'fasion-pairs-{self.phase}.csv')
        name_pairs = self.init_categories(pairLst)

        image_dir = os.path.join(root, f'{self.phase}_higher')
        bonesLst = os.path.join(root, f'fasion-annotation-{self.phase}.csv')
        canonical_dir = os.path.join(root, f'{self.phase}_higher_fix_canonical')
        return image_dir, canonical_dir, bonesLst, name_pairs

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
        P1_path = os.path.join(self.image_dir, P1_name) # person 1
        P2_path = os.path.join(self.image_dir, P2_name) # person 2
        Canonical_path = os.path.join(self.canonical_dir, f'{P1_name[:-4]}_2_{P2_name[:-4]}_vis.jpg')

        P1_img = Image.open(P1_path).convert('RGB')
        P2_img = Image.open(P2_path).convert('RGB')
        Canonical_img = Image.open(Canonical_path).convert('RGB')

        P1_img = F.resize(P1_img, self.load_size)
        P2_img = F.resize(P2_img, self.load_size)
        Canonical_img = F.resize(Canonical_img, self.load_size)

        # P1 preprocessing
        P1 = self.trans(P1_img)
        # BP1 = self.obtain_bone(P1_name)
        BP1 = torch.load(os.path.join(self.opt.dataroot, f'{self.phase}_map', P1_name.replace('jpg', 'pt')))[:18]
        # P2 preprocessing
        P2 = self.trans(P2_img)
        # BP2 = self.obtain_bone(P2_name)
        BP2 = torch.load(os.path.join(self.opt.dataroot, f'{self.phase}_map', P2_name.replace('jpg', 'pt')))[:18]
        # Canonical_img
        PC = self.trans(Canonical_img)
        # BPC = self.obtain_bone(None)
        BPC = torch.load(os.path.join(self.opt.dataroot, 'canonical_map.pt'))[:18]

        # self.check_bone_img_matching(P1, BP1, f'tmp/check_obtainbone/full_{index}_src.jpg')
        # self.check_bone_img_matching(P2, BP2, f'tmp/check_obtainbone/full_{index}_tgt.jpg')
        # self.check_bone_img_matching(PC, BPC, f'tmp/check_obtainbone/full_{index}_can.jpg')

        input_dict = {'src_image' : P1,
                      'src_map': BP1,
                      'tgt_image' : P2,
                      'tgt_map' : BP2,
                      'canonical_image' : PC,
                      'canonical_map' : BPC,
                      'path' : f'{P1_name.replace(".jpg", "")}_2_{P2_name.replace(".jpg", "")}_vis.jpg'}

        return input_dict


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

