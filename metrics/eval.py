import argparse
from networks import fid, inception, lpips, reconstruction
from networks import preprocess_path_for_deform_task
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='scripts to compute all statistics')
    parser.add_argument('--gt_path', help='Path to ground truth data', type=str)
    parser.add_argument('--distorated_path', help='Path to output data', type=str)
    parser.add_argument('--fid_real_path', help='Path to real images when calculate FID', type=str)
    parser.add_argument('--name', help='name of the experiment', type=str)
    parser.add_argument('--calculate_mask', action='store_true')
    parser.add_argument('--market', action='store_true')
    parser.add_argument('--old_size', type=int)
    parser.add_argument('--load_size', type=int)
    args = parser.parse_args()

    print('load start')

    fid = fid.FID()
    print('load FID')
    if args.market:
        rec = reconstruction.Reconstruction_Market_Metrics()
        print('load market rec')
    else:
        rec = reconstruction.Reconstruction_Metrics()
        print('load rec')

    lpips = lpips.LPIPS()
    print('load LPIPS')

    for arg in vars(args):
        print('[%s] =' % arg, getattr(args, arg))

    gt_list, distorated_list = preprocess_path_for_deform_task(args.gt_path, args.distorated_path)

    print('calculate LPIPS...')
    lpips_score = lpips.calculate_from_disk(distorated_list, gt_list, batch_size=64, sort=False)

    print('calculate fid metric...')
    fid_score = fid.calculate_from_disk(args.distorated_path, args.fid_real_path)

    print('calculate reconstruction metric...')
    rec_dic = rec.calculate_from_disk(distorated_list, gt_list, save_path=args.distorated_path, sort=False, debug=False)

    if args.calculate_mask:
        mask_lpips_score = lpips.calculate_mask_lpips(distorated_list, gt_list, sort=False)

    # dic = {}
    # dic['name'] = [args.name]
    # for key in rec_dic:
    #     dic[key] = rec_dic[key]
    # dic['fid'] = [fid_score]
    #
    # print('fid', fid_score)
    #
    # dic['lpips'] = [lpips_score]
    # print('lpips_score', lpips_score)
    #
    # if args.calculate_mask:
    #     dic['mask_lpips'] = [mask_lpips_score]