import argparse
import os

from metrics.networks import fid, inception, lpips, reconstruction
from metrics.networks import preprocess_path_for_deform_task, get_image_list
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch, gc
from metrics.networks.dataloader import MetricDataset
from metrics.networks import make_dataloader
def mean(buffer) :
    return sum(buffer) / len(buffer)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='scripts to compute all statistics')
    parser.add_argument('--gt_path', help='Path to ground truth data', type=str)
    parser.add_argument('--distorated_path', help='Path to output data', type=str)
    parser.add_argument('--fid_real_path', help='Path to real images when calculate FID', type=str)
    parser.add_argument('--name', help='name of the experiment', type=str)
    parser.add_argument('--calculate_mask', action='store_true')
    parser.add_argument('--market', action='store_true')
    parser.add_argument('--cv2', action='store_true')
    parser.add_argument('--gpu_id', type=int, default = 0)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--interpolation', type=str, default='bilinear')
    parser.set_defaults(old_size=(256, 256))
    parser.set_defaults(load_size=(256, 176)) # (h, w)
    args = parser.parse_args()
    for arg in vars(args):
        print('[%s] =' % arg, getattr(args, arg))
    print(f'Set GPU id : {args.gpu_id}')
    # device = torch.device(f"cuda:{args.gpu_id}")
    # torch.cuda.set_device(args.gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # Loading models
    fid = fid.FID()
    print('load FID')
    rec = reconstruction.Reconstruction_Market_Metrics() if args.market else reconstruction.Reconstruction_Metrics()
    print('load rec')
    lpips = lpips.LPIPS()
    print('load LPIPS')

    print('load start')
    gt_dict, distorated_dict = preprocess_path_for_deform_task(args.gt_path, args.distorated_path)

    score_dict = {'lpips' : [0] * 20,
                  'fid' : [0] * 20,
                  'ssim' : [0] * 20,
                  'ssim_256': [0] * 20,
                  'psnr' : [0] * 20,
                  'l1' : [0] * 20,
                  'mae' : [0] * 20}
    # Calculate FID from scratch
    fid_real_list = get_image_list(args.fid_real_path)
    dataloader = MetricDataset(args, fid_real_list, fid_real_list)
    dataloader1 = make_dataloader(dataloader, args.batchsize)
    fid_real_buffer = []
    for real_imgs, _, _ in tqdm(dataloader1, desc= 'Calculating FID real statics...') :
        real_imgs = real_imgs.cuda()
        with torch.no_grad():
            fid_score = fid.calculate_activation_statistics_of_images(real_imgs)
        fid_real_buffer.append(fid_score)
    gc.collect()
    torch.cuda.empty_cache()
    act = np.concatenate(fid_real_buffer, axis=0)
    m1 = np.mean(act, axis=0)
    s1 = np.cov(act, rowvar=False)

    # m1, s1 = fid.compute_statistics_of_path(args.fid_real_path, False)
    dict_scores = {'filename': [],
                   'lpips' : [],
                   'psnr' : [],
                   'ssim' : []}
    for (num_keypoint, gt_list), (num_keypoint, distorated_list) in zip(gt_dict.items(), distorated_dict.items()) :
        if num_keypoint < 19: continue
        dataloader = MetricDataset(args, gt_list, distorated_list)
        dataloader1 = make_dataloader(dataloader, args.batchsize)

        # score buffers
        lpips_buffers = []
        fid_buffers = []
        ssim_buffers = []
        ssim_256_buffers = []
        psnr_buffers = []
        l1_buffers = []
        mae_buffers = []

        for gt_imgs, distorated_imgs, gt_path in tqdm(dataloader1, desc = f'Calculating {num_keypoint} key point scores') :
            gt_imgs = gt_imgs.cuda()
            distorated_imgs = distorated_imgs.cuda()
            with torch.no_grad() :
                lpips_score = lpips(distorated_imgs, gt_imgs)
                rec_dict = rec(distorated_imgs, gt_imgs)
                fid_score = fid.calculate_activation_statistics_of_images(distorated_imgs)
            dict_scores['lpips'].extend(lpips_score.squeeze().cpu().tolist())
            dict_scores['filename'].extend(gt_path)
            dict_scores['psnr'].extend(rec_dict['psnr'])
            dict_scores['ssim'].extend(rec_dict['ssim'])

            lpips_buffers.append(lpips_score.mean().item())
            fid_buffers.append(fid_score)

            ssim_buffers.extend(rec_dict['ssim'])
            ssim_256_buffers.extend(rec_dict['ssim_256'])
            psnr_buffers.extend(rec_dict['psnr'])
            l1_buffers.extend(rec_dict['l1'])
            mae_buffers.extend(rec_dict['mae'])

            gc.collect()
            torch.cuda.empty_cache()

        act = np.concatenate(fid_buffers, axis = 0)
        m2 = np.mean(act, axis=0)
        s2 = np.cov(act, rowvar=False)
        score_dict['fid'][num_keypoint] = fid.calculate_frechet_distance(m1, s1, m2, s2)
        score_dict['lpips'][num_keypoint] = mean(lpips_buffers)
        score_dict['ssim'][num_keypoint] = mean(ssim_buffers)
        score_dict['ssim_256'][num_keypoint] = mean(ssim_256_buffers)
        score_dict['psnr'][num_keypoint] = mean(psnr_buffers)
        score_dict['l1'][num_keypoint] = mean(l1_buffers)
        score_dict['mae'][num_keypoint] = mean(mae_buffers)

    df_score = pd.DataFrame.from_dict(score_dict).T
    df_score = df_score[[19]+[x for x in range(7, 19)]]
    df_score.rename(columns = {19 : 'Total'}, inplace=True)
    df_score.to_csv(f'./eval_results/{args.name}.csv')

    pd.DataFrame.from_dict(dict_scores).to_csv(f'./eval_results/{args.name}_rank.csv', index=False)
    #
    # fid_score = fid.calculate_from_disk(args.distorated_path, args.fid_real_path)
    # print(f'My FID score {df_score["Total"]["fid"]}')