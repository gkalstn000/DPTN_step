import torch
import torch.utils.data as data
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from imageio import imread



interpolation_dict = {'bilinear' : Image.BILINEAR,
                      'nearest' : Image.NEAREST,
                      'bicubic' : Image.BICUBIC,
                      'area': None}
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
        self.interpolation = interpolation_dict[opt.interpolation]
    def __getitem__(self, index):
        gt_path = self.gt_list[index]
        distorated_path = self.distorated_list[index]

        # gt_image = Image.open(gt_path).convert('RGB')
        # distorated_image = Image.open(distorated_path).convert('RGB')
        #
        # gt_image = F.resize(gt_image, self.opt.load_size, self.interpolation)
        # distorated_image = F.resize(distorated_image, self.opt.load_size, self.interpolation)
        #
        # return self.trans(gt_image).float(), self.trans(distorated_image).float()

        gt_image = self.cv2_loading(gt_path)
        distorated_image = self.cv2_loading(distorated_path)
        return torch.Tensor(gt_image), torch.Tensor(distorated_image)

    def cv2_loading(self, image_path):
        interpolation_dict_ = {'bilinear' : cv2.INTER_LINEAR,
                              'nearest' : cv2.INTER_NEAREST,
                              'bicubic' : cv2.INTER_CUBIC,
                              'area' : cv2.INTER_AREA}
        img_array = imread(image_path)
        img_array = cv2.resize(img_array, self.opt.load_size[::-1], interpolation_dict_[self.opt.interpolation]).astype(np.float32)
        img_array = img_array.transpose((2, 0, 1)) / 255.0
        return img_array
    def __len__(self):
        return len(self.gt_list)


