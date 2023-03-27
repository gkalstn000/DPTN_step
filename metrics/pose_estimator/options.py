import argparse
import os
import util.util as util
import torch
def option():
    """
        Define args that is used in project
    """
    parser = argparse.ArgumentParser(description="Pose guided image generation usign deformable skip layers")
    parser.add_argument("--id", default='DPTN_higher', help="experiment result's name")
    parser.add_argument("--batch_size", default=4, type=int, help='Size of the batch')
    parser.add_argument("--gpu_id", default=1, type=int, help='GPU ID')


    parser.add_argument('--dataset', default='fashion', choices=['market', 'fashion'],
                        help='Market or fasion')
    parser.add_argument("--pose_estimator", default='metrics/pose_estimator/pose_model.pth',
                            help='Pretrained model for cao pose estimator')

    opt = parser.parse_args()

    dataroot = '/home/red/external/msha/datasets'

    opt.images_dir_train = os.path.join(dataroot, opt.dataset, 'train')
    opt.images_dir_test = os.path.join(dataroot, opt.dataset, 'test')

    opt.annotations_file_train = os.path.join(dataroot, opt.dataset, f'{opt.dataset}-annotation-train.csv')
    opt.annotations_file_test = os.path.join(dataroot, opt.dataset, f'{opt.dataset}-annotation-test.csv')

    opt.pairs_file_train = os.path.join(dataroot, opt.dataset, f'{opt.dataset}-pairs-train.csv')
    opt.pairs_file_test = os.path.join(dataroot, opt.dataset, f'{opt.dataset}-pairs-test.csv')

    output_path = os.path.join('eval_results', 'pose_estimate')
    util.mkdirs(output_path)

    opt.generated_path = os.path.join('results', opt.id)
    opt.output_file = os.path.join(output_path, opt.id+'.csv')

    if opt.dataset == 'fashion':
        opt.image_size = (256, 256)
    else:
        opt.image_size = (128, 64)
    

    del opt.dataset

    torch.cuda.set_device(opt.gpu_id)

    return opt
