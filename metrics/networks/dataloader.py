import torch
import torch.utils.data as data
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from imageio import imread

class MetricDataset(data.Dataset) :
    def __init__(self, opt, gt_list, distorated_list):
        super(MetricDataset, self).__init__()
        self.gt_list = gt_list
        self.distorated_list = distorated_list
        self.opt = opt
        transform_list = []
        transform_list.append(transforms.ToTensor())
        # transform_list.append(transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)))
        self.trans = transforms.Compose(transform_list)
    def __getitem__(self, index):
        gt_path = self.gt_list[index]
        distorated_path = self.distorated_list[index]

        gt_image = Image.open(gt_path).convert('RGB')
        distorated_image = Image.open(distorated_path).convert('RGB')

        gt_image = F.resize(gt_image, self.opt.load_size)
        distorated_image = F.resize(distorated_image, self.opt.load_size)

        return self.trans(gt_image), self.trans(distorated_image)

        # gt_image = self.cv2_loading(gt_path)
        # distorated_image = self.cv2_loading(distorated_path)
        # return torch.Tensor(gt_image), torch.Tensor(distorated_image)

    def cv2_loading(self, image_path):
        img_array = imread(image_path)
        img_array = cv2.resize(img_array, self.opt.load_size[::-1]).astype(np.float32)
        img_array = img_array.transpose((2, 0, 1)) / 255
        return img_array
    def __len__(self):
        return len(self.gt_list)


