from ..PerceptualSimilarity.models import dist_model as dm
from . import get_image_list, create_masked_image
import numpy as np
import torch
from imageio import imread
import pandas as pd
from tqdm import tqdm, trange
import torch.nn.functional as nnf
from PIL import Image

class LPIPS():
    def __init__(self, use_gpu=True):
        self.model = dm.DistModel()
        self.model.initialize(model='net-lin', net='alex',use_gpu=use_gpu)
        self.use_gpu=use_gpu

    def __call__(self, image_1, image_2):
        """
            image_1: images with size (n, 3, w, h) with value [-1, 1]
            image_2: images with size (n, 3, w, h) with value [-1, 1]
        """
        result = self.model.forward(image_1, image_2)
        return result

    def calculate_from_disk(self, path_1, path_2, batch_size=1, verbose=False, sort=True):
        if sort:
            files_1 = sorted(get_image_list(path_1))
            files_2 = sorted(get_image_list(path_2))
        else:
            files_1 = get_image_list(path_1)
            files_2 = get_image_list(path_2)


        imgs_1 = np.array([np.array(Image.open(str(fn)).resize((176, 256), Image.BICUBIC)).astype(np.float32)/127.5-1 for fn in tqdm(files_1, desc='load gen files')])
        imgs_2 = np.array([np.array(Image.open(str(fn)).resize((176, 256), Image.BICUBIC)).astype(np.float32)/127.5-1 for fn in tqdm(files_2, desc='load gt files')])

        # Bring images to shape (B, 3, H, W)
        imgs_1 = imgs_1.transpose((0, 3, 1, 2))
        imgs_2 = imgs_2.transpose((0, 3, 1, 2))

        result=[]


        d0 = imgs_1.shape[0]
        if batch_size > d0:
            print(('Warning: batch size is bigger than the data size. '
                   'Setting batch size to data size'))
            batch_size = d0

        n_batches = d0 // batch_size

        for i in trange(n_batches):
            if verbose:
                print('\rPropagating batch %d/%d' % (i + 1, n_batches))
            start = i * batch_size
            end = start + batch_size

            img_1_batch = torch.from_numpy(imgs_1[start:end]).type(torch.FloatTensor)
            img_2_batch = torch.from_numpy(imgs_2[start:end]).type(torch.FloatTensor)

            if self.use_gpu:
                img_1_batch = img_1_batch.cuda()
                img_2_batch = img_2_batch.cuda()
            # print(img_1_batch.size())
            if img_1_batch.size()[2:] != (256, 176) :
                img_1_batch = nnf.interpolate(img_1_batch, size=(256, 176), mode='bicubic', align_corners=False)
            if img_2_batch.size()[2:] != (256, 176) :
                img_2_batch = nnf.interpolate(img_2_batch, size=(256, 176), mode='bicubic', align_corners=False)
            a = self.model.forward(img_1_batch, img_2_batch).item()
            result.append(a)


        distance = np.average(result)
        print('lpips: ', distance)
        return distance

    def calculate_mask_lpips(self, path_1, path_2, batch_size=64, verbose=False, sort=True):
        if sort:
            files_1 = sorted(get_image_list(path_1))
            files_2 = sorted(get_image_list(path_2))
        else:
            files_1 = get_image_list(path_1)
            files_2 = get_image_list(path_2)

        imgs_1=[]
        imgs_2=[]
        bonesLst = '/media/data1/zhangpz/DataSet/Market/market-annotation-test.csv'
        annotation_file = pd.read_csv(bonesLst, sep=':')
        annotation_file = annotation_file.set_index('name')

        for i in range(len(files_1)):
            string = annotation_file.loc[os.path.basename(files_2[i])]
            mask = np.tile(np.expand_dims(create_masked_image(string).astype(np.float32), -1), (1,1,3))#.repeat(1,1,3)
            imgs_1.append((imread(str(files_1[i])).astype(np.float32)/127.5-1)*mask)
            imgs_2.append((imread(str(files_2[i])).astype(np.float32)/127.5-1)*mask)

        # Bring images to shape (B, 3, H, W)
        imgs_1 = np.array(imgs_1)
        imgs_2 = np.array(imgs_2)
        imgs_1 = imgs_1.transpose((0, 3, 1, 2))
        imgs_2 = imgs_2.transpose((0, 3, 1, 2))

        result=[]


        d0 = imgs_1.shape[0]
        if batch_size > d0:
            print(('Warning: batch size is bigger than the data size. '
                   'Setting batch size to data size'))
            batch_size = d0

        n_batches = d0 // batch_size

        for i in range(n_batches):
            if verbose:
                print('\rPropagating batch %d/%d' % (i + 1, n_batches))
            start = i * batch_size
            end = start + batch_size

            img_1_batch = torch.from_numpy(imgs_1[start:end]).type(torch.FloatTensor)
            img_2_batch = torch.from_numpy(imgs_2[start:end]).type(torch.FloatTensor)

            if self.use_gpu:
                img_1_batch = img_1_batch.cuda()
                img_2_batch = img_2_batch.cuda()


            result.append(self.model.forward(img_1_batch, img_2_batch))


        distance = torch.mean(torch.stack(result))
        print('lpips_mask: ', distance)
        return distance