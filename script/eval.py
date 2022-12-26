import argparse
import sys, os
from metrics import FID,Reconstruction_Market_Metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='script to compute all statistics')
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

    fid = FID()
    if args.market:
        rec = Reconstruction_Market_Metrics()
        print('load market rec')
    else:
        rec = Reconstruction_Metrics()
        print('load rec')