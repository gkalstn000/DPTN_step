import os
import numpy as np
from script.metrics.fid import FID

import glob
import argparse
import json
from skimage.draw import disk, polygon


def pad_256(img):
    result = np.ones((256, 256, 3), dtype=float) * 255
    result[:,40:216,:] = img
    return result


def get_image_list(flist):
    if isinstance(flist, list):
        return flist

    # flist: image file path, image directory path, text file flist path
    if isinstance(flist, str):
        if os.path.isdir(flist):
            flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
            flist.sort()
            return flist

        if os.path.isfile(flist):
            try:
                return np.genfromtxt(flist, dtype=np.str)
            except:
                return [flist]
    print('can not read files from %s return empty list'%flist)
    return []


def compare_l1(img_true, img_test):
    img_true = img_true.astype(np.float32)
    img_test = img_test.astype(np.float32)
    return np.mean(np.abs(img_true - img_test))    


def compare_mae(img_true, img_test):
    img_true = img_true.astype(np.float32)
    img_test = img_test.astype(np.float32)
    return np.sum(np.abs(img_true - img_test)) / np.sum(img_true + img_test)


def preprocess_path_for_deform_task(gt_path, distorted_path):
    distorted_image_list = sorted(get_image_list(distorted_path))
    gt_list=[]
    distorated_list=[]

    for distorted_image in distorted_image_list:
        image = os.path.basename(distorted_image)
        image = image.split('_2_')[-1]
        image = image.split('_vis')[0] +'.jpg'
        gt_image = os.path.join(gt_path, image)
        if not os.path.isfile(gt_image):
            print("hhhhhhhhh")
            print(gt_image)
            continue
        gt_list.append(gt_image)
        distorated_list.append(distorted_image)    

    return gt_list, distorated_list


def produce_ma_mask(kp_array, img_size=(128, 64), point_radius=4):
    MISSING_VALUE = -1
    from skimage.morphology import dilation, erosion, square
    mask = np.zeros(shape=img_size, dtype=bool)
    limbs = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10],
              [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17],
               [1,16], [16,18], [2,17], [2,18], [9,12], [12,6], [9,3], [17,18]]
    limbs = np.array(limbs) - 1
    for f, t in limbs:
        from_missing = kp_array[f][0] == MISSING_VALUE or kp_array[f][1] == MISSING_VALUE
        to_missing = kp_array[t][0] == MISSING_VALUE or kp_array[t][1] == MISSING_VALUE
        if from_missing or to_missing:
            continue

        norm_vec = kp_array[f] - kp_array[t]
        norm_vec = np.array([-norm_vec[1], norm_vec[0]])
        norm_vec = point_radius * norm_vec / np.linalg.norm(norm_vec)


        vetexes = np.array([
            kp_array[f] + norm_vec,
            kp_array[f] - norm_vec,
            kp_array[t] - norm_vec,
            kp_array[t] + norm_vec
        ])
        yy, xx = polygon(vetexes[:, 0], vetexes[:, 1], shape=img_size)
        mask[yy, xx] = True

    for i, joint in enumerate(kp_array):
        if kp_array[i][0] == MISSING_VALUE or kp_array[i][1] == MISSING_VALUE:
            continue
        yy, xx = disk((joint[0], joint[1]), radius=point_radius, shape=img_size)
        mask[yy, xx] = True

    mask = dilation(mask, square(5))
    mask = erosion(mask, square(5))
    return mask 


def load_pose_cords_from_strings(y_str, x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    return np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)       


def create_masked_image(ano_to):
    kp_to = load_pose_cords_from_strings(ano_to['keypoints_y'], ano_to['keypoints_x'])
    mask = produce_ma_mask(kp_to)
    return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='script to compute all statistics')
    parser.add_argument('--gt_path', help='Path to ground truth data', type=str)
    parser.add_argument('--distorated_path', help='Path to output data', type=str)
    parser.add_argument('--fid_real_path', help='Path to real images when calculate FID', type=str)
    parser.add_argument('--name', help='name of the experiment', type=str)
    parser.add_argument('--calculate_mask', action='store_true')
    parser.add_argument('--market', action='store_true')
    args = parser.parse_args()

    print('load start')
    for arg in vars(args):
        print('[%s] =' % arg, getattr(args, arg))
    gt_list, distorated_list = preprocess_path_for_deform_task(args.gt_path, args.distorated_path)


    print('load FID')
    fid = FID()

    print('calculate fid metric...')
    fid_score = fid.calculate_from_disk(args.distorated_path, args.fid_real_path)

    dic = {}
    dic['fid'] = [fid_score]
    print('fid', fid_score)





    #
    # if args.market:
    #     rec = Reconstruction_Market_Metrics()
    #     print('load market rec')
    # else:
    #     rec = Reconstruction_Metrics()
    #     print('load rec')
    #
    # lpips = LPIPS()
    # print('load LPIPS')
    #
    # print('calculate LPIPS...')
    # lpips_score = lpips.calculate_from_disk(distorated_list, gt_list, sort=False)

    #
    # print('calculate reconstruction metric...')
    # rec_dic = rec.calculate_from_disk(distorated_list, gt_list, save_path=args.distorated_path, sort=False, debug=False)
    #
    # if args.calculate_mask:
    #     mask_lpips_score = lpips.calculate_mask_lpips(distorated_list, gt_list, sort=False)

    # dic['name'] = [args.name]
    # for key in rec_dic:
    #     dic[key] = rec_dic[key]
    #
    #
    #
    # dic['lpips']=[lpips_score]
    # print('lpips_score', lpips_score)
    #
    # if args.calculate_mask:
    #     dic['mask_lpips']=[mask_lpips_score]








