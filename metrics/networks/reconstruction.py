from . import get_image_list, compare_l1, compare_mae, pad_256
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import torch
import matplotlib.pyplot as plt
from imageio import imread
from PIL import Image
import numpy as np
import os


class Reconstruction_Metrics():
    def __init__(self, metric_list=['ssim', 'ssim_256', 'psnr', 'l1', 'mae'], data_range=1, win_size=51, multichannel=True):
        self.data_range = data_range
        self.win_size = win_size
        self.multichannel = multichannel
        for metric in metric_list:
            if metric in ['ssim', 'ssim_256','psnr', 'l1', 'mae']:
                setattr(self, metric, True)
            else:
                print('unsupport reconstruction metric: %s ' %metric)

    def __call__(self, inputs, gts, padding=True):
        """
        inputs: the generated image, size (b,c,w,h), data range(0, data_range)
        gts:    the ground-truth image, size (b,c,w,h), data range(0, data_range)
        """
        result = dict()
        [b ,c ,h ,w] = inputs.size()
        if padding:
            padding = torch.ones([b, c, h, 40])
            inputs = torch.concatenate([padding, inputs.cpu(), padding], dim = 3).numpy().transpose(0, 2 ,3 ,1)
            gts = torch.concatenate([padding, gts.cpu(), padding], dim=3).numpy().transpose(0, 2 ,3 ,1)
        else :
            inputs = inputs.cpu().numpy().transpose(0, 2 ,3 ,1)
            gts = gts.cpu().numpy().transpose(0, 2 ,3 ,1)

        # inputs = inputs.view( b *n, h, w).detach().cpu().numpy().astype(np.float32).transpose(1 ,2 ,0)
        # gts = gts.view( b *n, h, w).detach().cpu().numpy().astype(np.float32).transpose(1 ,2 ,0)
        psnr = []
        ssim = []
        ssim_256 = []
        mae = []
        l1 = []
        for gt, input in zip(gts, inputs) :
            if hasattr(self, 'ssim'):
                ssim_value = compare_ssim(gt, input, data_range=self.data_range,
                                          win_size=self.win_size, multichannel=self.multichannel)
                ssim.append(ssim_value)

                gt_256 = gt * 255.0
                input_256 = input * 255.0
                ssim_256_value = compare_ssim(gt_256, input_256, gaussian_weights=True, sigma=1.5,
                                                 use_sample_covariance=False, multichannel=True,
                                                 data_range=255)
                ssim_256.append(ssim_256_value)

            if hasattr(self, 'psnr'):
                psnr_value = compare_psnr(gt, input, data_range=self.data_range)
                psnr.append(psnr_value)

            if hasattr(self, 'l1'):
                l1_value = compare_l1(gt, input)
                l1.append(l1_value)

            if hasattr(self, 'mae'):
                mae_value = compare_mae(gt, input)
                mae.append(mae_value)
        result['ssim'] = round(np.mean(ssim), 4)
        result['ssim_256'] = round(np.mean(ssim_256), 4)
        result['psnr'] = round(np.mean(psnr), 4)
        result['l1'] = round(np.mean(l1), 4)
        result['mae'] = round(np.mean(mae), 4)
        return result

    def calculate_from_disk(self, inputs, gts, save_path=None, sort=True, debug=0):
        """
            inputs: .txt files, floders, image files (string), image files (list)
            gts: .txt files, floders, image files (string), image files (list)
        """
        if sort:
            input_image_list = sorted(get_image_list(inputs))
            gt_image_list = sorted(get_image_list(gts))
        else:
            input_image_list = get_image_list(inputs)
            gt_image_list = get_image_list(gts)
        npz_file = os.path.join(save_path, 'metrics.npz')
        if os.path.exists(npz_file):
            f = np.load(npz_file)
            psnr ,ssim ,ssim_256 ,mae ,l1 =f['psnr'] ,f['ssim'] ,f['ssim_256'] ,f['mae'] ,f['l1']
        else:
            psnr = []
            ssim = []
            ssim_256 = []
            mae = []
            l1 = []
            names = []

            for index in range(len(input_image_list)):
                name = os.path.basename(input_image_list[index])
                names.append(name)

                img_gt = pad_256(Image.open(str(gt_image_list[index]))).astype(np.float32) / 255.0
                img_pred = pad_256(Image.open(str(input_image_list[index]))).astype(np.float32) / 255.0


                if debug != 0:
                    plt.subplot('121')
                    plt.imshow(img_gt)
                    plt.title('Groud truth')
                    plt.subplot('122')
                    plt.imshow(img_pred)
                    plt.title('Output')
                    plt.show()

                psnr.append(compare_psnr(img_gt, img_pred, data_range=self.data_range))
                ssim.append(compare_ssim(img_gt, img_pred, data_range=self.data_range,
                                         win_size=self.win_size ,multichannel=self.multichannel))
                mae.append(compare_mae(img_gt, img_pred))
                l1.append(compare_l1(img_gt, img_pred))

                img_gt_256 = img_gt *255.0
                img_pred_256 = img_pred *255.0
                ssim_256.append(compare_ssim(img_gt_256, img_pred_256, gaussian_weights=True, sigma=1.5,
                                             use_sample_covariance=False, multichannel=True,
                                             data_range=img_pred_256.max() - img_pred_256.min()))
                if np.mod(index, 200) == 0:
                    print(
                        str(index) + ' images processed',
                        "PSNR: %.4f" % round(np.mean(psnr), 4),
                        "SSIM: %.4f" % round(np.mean(ssim), 4),
                        "SSIM_256: %.4f" % round(np.mean(ssim_256), 4),
                        "MAE: %.4f" % round(np.mean(mae), 4),
                        "l1: %.4f" % round(np.mean(l1), 4),
                        )

            if save_path:
                np.savez(save_path + '/metrics.npz', psnr=psnr, ssim=ssim, ssim_256=ssim_256, mae=mae, l1=l1, names=names)

        print(
            "PSNR: %.4f" % round(np.mean(psnr), 4),
            "PSNR Variance: %.4f" % round(np.var(psnr), 4),
            "SSIM: %.4f" % round(np.mean(ssim), 4),
            "SSIM Variance: %.4f" % round(np.var(ssim), 4),
            "SSIM_256: %.4f" % round(np.mean(ssim_256), 4),
            "SSIM_256 Variance: %.4f" % round(np.var(ssim_256), 4),
            "MAE: %.4f" % round(np.mean(mae), 4),
            "MAE Variance: %.4f" % round(np.var(mae), 4),
            "l1: %.4f" % round(np.mean(l1), 4),
            "l1 Variance: %.4f" % round(np.var(l1), 4)
        )

        dic = {"psnr" :[round(np.mean(psnr), 6)],
               "psnr_variance": [round(np.var(psnr), 6)],
               "ssim": [round(np.mean(ssim), 6)],
               "ssim_variance": [round(np.var(ssim), 6)],
               "ssim_256": [round(np.mean(ssim_256), 6)],
               "ssim_256_variance": [round(np.var(ssim_256), 6)],
               "mae": [round(np.mean(mae), 6)],
               "mae_variance": [round(np.var(mae), 6)],
               "l1": [round(np.mean(l1), 6)],
               "l1_variance": [round(np.var(l1), 6)] }

        return dic


class Reconstruction_Market_Metrics():
    def __init__(self, metric_list=['ssim', 'psnr', 'l1', 'mae'], data_range=1, win_size=51, multichannel=True):
        self.data_range = data_range
        self.win_size = win_size
        self.multichannel = multichannel
        for metric in metric_list:
            if metric in ['ssim', 'psnr', 'l1', 'mae']:
                setattr(self, metric, True)
            else:
                print('unsupport reconstruction metric: %s' % metric)

    def __call__(self, inputs, gts):
        """
        inputs: the generated image, size (b,c,w,h), data range(0, data_range)
        gts:    the ground-truth image, size (b,c,w,h), data range(0, data_range)
        """
        result = dict()
        [b, n, w, h] = inputs.size()
        inputs = inputs.view(b * n, w, h).detach().cpu().numpy().astype(np.float32).transpose(1, 2, 0)
        gts = gts.view(b * n, w, h).detach().cpu().numpy().astype(np.float32).transpose(1, 2, 0)

        if hasattr(self, 'ssim'):
            ssim_value = compare_ssim(inputs, gts, data_range=self.data_range,
                                      win_size=self.win_size, multichannel=self.multichannel)
            result['ssim'] = ssim_value

        if hasattr(self, 'psnr'):
            psnr_value = compare_psnr(inputs, gts, self.data_range)
            result['psnr'] = psnr_value

        if hasattr(self, 'l1'):
            l1_value = compare_l1(inputs, gts)
            result['l1'] = l1_value

        if hasattr(self, 'mae'):
            mae_value = compare_mae(inputs, gts)
            result['mae'] = mae_value
        return result

    def calculate_from_disk(self, inputs, gts, save_path=None, sort=True, debug=0):
        """
            inputs: .txt files, floders, image files (string), image files (list)
            gts: .txt files, floders, image files (string), image files (list)
        """
        if sort:
            input_image_list = sorted(get_image_list(inputs))
            gt_image_list = sorted(get_image_list(gts))
        else:
            input_image_list = get_image_list(inputs)
            gt_image_list = get_image_list(gts)
        npz_file = os.path.join(save_path, 'metrics.npz')
        if os.path.exists(npz_file):
            f = np.load(npz_file)
            psnr, ssim, ssim_256, mae, l1 = f['psnr'], f['ssim'], f['ssim_256'], f['mae'], f['l1']
        else:
            psnr = []
            ssim = []
            ssim_256 = []
            mae = []
            l1 = []
            names = []

            for index in range(len(input_image_list)):
                name = os.path.basename(input_image_list[index])
                names.append(name)

                img_gt = imread(str(gt_image_list[index])).astype(np.float32) / 255.0
                img_pred = imread(str(input_image_list[index])).astype(np.float32) / 255.0

                if debug != 0:
                    plt.subplot('121')
                    plt.imshow(img_gt)
                    plt.title('Groud truth')
                    plt.subplot('122')
                    plt.imshow(img_pred)
                    plt.title('Output')
                    plt.show()

                psnr.append(compare_psnr(img_gt, img_pred, data_range=self.data_range))
                ssim.append(compare_ssim(img_gt, img_pred, data_range=self.data_range,
                                         win_size=self.win_size, multichannel=self.multichannel))
                mae.append(compare_mae(img_gt, img_pred))
                l1.append(compare_l1(img_gt, img_pred))

                img_gt_256 = img_gt * 255.0
                img_pred_256 = img_pred * 255.0
                ssim_256.append(compare_ssim(img_gt_256, img_pred_256, gaussian_weights=True, sigma=1.5,
                                             use_sample_covariance=False, multichannel=True,
                                             data_range=img_pred_256.max() - img_pred_256.min()))
                if np.mod(index, 200) == 0:
                    print(
                        str(index) + ' images processed',
                        "PSNR: %.4f" % round(np.mean(psnr), 4),
                        "SSIM: %.4f" % round(np.mean(ssim), 4),
                        "SSIM_256: %.4f" % round(np.mean(ssim_256), 4),
                        "MAE: %.4f" % round(np.mean(mae), 4),
                        "l1: %.4f" % round(np.mean(l1), 4),
                        )

            if save_path:
                np.savez(save_path + '/metrics.npz', psnr=psnr, ssim=ssim, ssim_256=ssim_256, mae=mae, l1=l1,
                         names=names)

        print(
            "PSNR: %.4f" % round(np.mean(psnr), 4),
            "PSNR Variance: %.4f" % round(np.var(psnr), 4),
            "SSIM: %.4f" % round(np.mean(ssim), 4),
            "SSIM Variance: %.4f" % round(np.var(ssim), 4),
            "SSIM_256: %.4f" % round(np.mean(ssim_256), 4),
            "SSIM_256 Variance: %.4f" % round(np.var(ssim_256), 4),
            "MAE: %.4f" % round(np.mean(mae), 4),
            "MAE Variance: %.4f" % round(np.var(mae), 4),
            "l1: %.4f" % round(np.mean(l1), 4),
            "l1 Variance: %.4f" % round(np.var(l1), 4)
        )

        dic = {"psnr": [round(np.mean(psnr), 6)],
               "psnr_variance": [round(np.var(psnr), 6)],
               "ssim": [round(np.mean(ssim), 6)],
               "ssim_variance": [round(np.var(ssim), 6)],
               "ssim_256": [round(np.mean(ssim_256), 6)],
               "ssim_256_variance": [round(np.var(ssim_256), 6)],
               "mae": [round(np.mean(mae), 6)],
               "mae_variance": [round(np.var(mae), 6)],
               "l1": [round(np.mean(l1), 6)],
               "l1_variance": [round(np.var(l1), 6)]}

        return dic