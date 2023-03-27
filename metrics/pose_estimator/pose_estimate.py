import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

import skimage.transform as st
from skimage.io import imread
from skimage.transform import resize

from metrics.pose_estimator.options import option
from metrics.pose_estimator.functions import compute_cordinates
from metrics.pose_estimator.lib.network.rtpose_vgg import get_model
from metrics.pose_estimator.evaluate.coco_eval import get_outputs
opt = option()
model = get_model('vgg19')
model.load_state_dict(torch.load(opt.pose_estimator))
model = model.cuda()
model.float()
model.eval()

# input_folder = './checkpoints/PoseTransfer_market/output'
# output_path = './checkpoints/PoseTransfer_market/output_pckh.csv'

# input_folder = './checkpoints/PoseTransfer_deepfashion/output'
# output_path = './checkpoints/PoseTransfer_deepfashion/output_pckh.csv'


img_list = os.listdir(opt.generated_path)

threshold = 0.1
boxsize = 368
scale_search = [0.5, 1, 1.5, 2]

if os.path.exists(opt.output_file):
    df = pd.read_csv(opt.output_file, sep = ':')
else:
    init_data = {'name':[],
                 'keypoints_y':[],
                 'keypoints_x':[]}
    df = pd.DataFrame(init_data)

processed_names = df.name.to_list()

multiplier = [x * boxsize / opt.image_size[0] for x in scale_search]

# for image_name in tqdm(os.listdir(input_folder)):
for image_name in tqdm(img_list):
    if image_name in processed_names:
        continue

    image_array = imread(os.path.join(opt.generated_path, image_name))[:, :, ::-1]  # B,G,R order
    heatmap_avg = np.zeros((opt.image_size[0], opt.image_size[1], 19))
    paf_avg = np.zeros((opt.image_size[0], opt.image_size[1], 38))

    for scale in multiplier :
        new_size = (np.array(image_array.shape[:2]) * scale).astype(np.int32)
        imageToTest = resize(image_array, new_size, order=3, preserve_range=True)
        imageToTest_padded = imageToTest[np.newaxis, :, :, :]/255 - 0.5
        imageToTest_padded = np.transpose(imageToTest_padded, (0, 3, 1, 2))
        with torch.no_grad() :
            predicted_outputs, _ = model(torch.Tensor(imageToTest_padded).float().cuda())

            output1, output2 = predicted_outputs[-2], predicted_outputs[-1]
            heatmap = output2.cpu().data.numpy().transpose(0, 2, 3, 1)[0]
            paf = output1.cpu().data.numpy().transpose(0, 2, 3, 1)[0]

        heatmap_avg += heatmap
        paf_avg += paf

    heatmap_avg /= len(multiplier)

    pose_cords = compute_cordinates(heatmap_avg, paf_avg)

    print( "%s: %s: %s" % (image_name, str(list(pose_cords[:, 0])), str(list(pose_cords[:, 1]))))
