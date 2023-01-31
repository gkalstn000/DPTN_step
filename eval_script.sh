# 저해상도
#python eval.py --gt_path ./datasets/fashion/test --fid_real_path ./datasets/fashion/train --distorated_path ./results/DPTN_fashion --name DPTN
#python eval.py --gt_path ./datasets/fashion/test --fid_real_path ./datasets/fashion/train --distorated_path ./results/DPTN_2_stage --name DPTN_2_stage
#python eval.py --gt_path ./datasets/fashion/test --fid_real_path ./datasets/fashion/train --distorated_path ./results/DPTN_low_canonical/test_latest/images/synthesized_target_image --name DPTN_2_stage_train
#
#python eval.py --gt_path ./datasets/fashion/test --fid_real_path ./datasets/fashion/train --distorated_path ./results/s_to_s/test_latest/images/synthesized_target_image --name s_to_s
#python eval.py --gt_path ./datasets/fashion/test --fid_real_path ./datasets/fashion/train --distorated_path ./results/c_to_c/test_latest/images/synthesized_target_image --name c_to_c
#python eval.py --gt_path ./datasets/fashion/test --fid_real_path ./datasets/fashion/train --distorated_path ./results/c_to_s/test_latest/images/synthesized_target_image --name c_to_s
#python eval.py --gt_path ./datasets/fashion/test --fid_real_path ./datasets/fashion/train --distorated_path ./results/c2c_spade_dec/test_latest/images/synthesized_target_image --name c2c_spade_dec
#
#
## 고해상도
#python eval.py --gt_path ./datasets/fashion/test_higher --fid_real_path ./datasets/fashion/train_higher --distorated_path ./results/DPTN_higher/test_latest/images/synthesized_target_image --name DPTN_higher
#
## 고해상도 gen 저해상도 gt
#python eval.py --gt_path ./datasets/fashion/test --fid_real_path ./datasets/fashion/train --distorated_path ./results/DPTN_higher/test_latest/images/synthesized_target_image --name DPTN_higher_with_lower
#
## 고해상도 gt 저해상도 gen
#python eval.py --gt_path ./datasets/fashion/test_higher --fid_real_path ./datasets/fashion/train_higher --distorated_path ./results/DPTN_fashion --name DPTN_lower_with_higher

#python eval.py --gt_path ./datasets/fashion/test --fid_real_path ./datasets/fashion/train --distorated_path ./results/DPTN_fashion --gpu_id 0 --name DPTN_fashion_cv2_bilinear --interpolation bilinear
#python eval.py --gt_path ./datasets/fashion/test --fid_real_path ./datasets/fashion/train --distorated_path ./results/DPTN_fashion --gpu_id 0 --name DPTN_fashion_cv2_nearest --interpolation nearest
#python eval.py --gt_path ./datasets/fashion/test --fid_real_path ./datasets/fashion/train --distorated_path ./results/DPTN_fashion --gpu_id 0 --name DPTN_fashion_cv2_bicubic --interpolation bicubic
#python eval.py --gt_path ./datasets/fashion/test --fid_real_path ./datasets/fashion/train --distorated_path ./results/DPTN_fashion --gpu_id 0 --name DPTN_fashion_cv2_area --interpolation area
#python eval.py --gt_path ./datasets/fashion/test --fid_real_path ./datasets/fashion/train --distorated_path ./results/DPTN_fashion --gpu_id 0 --name DPTN_fashion_pil_bilinear --interpolation bilinear
#python eval.py --gt_path ./datasets/fashion/test --fid_real_path ./datasets/fashion/train --distorated_path ./results/DPTN_fashion --gpu_id 0 --name DPTN_fashion_pil_nearest --interpolation nearest
#python eval.py --gt_path ./datasets/fashion/test --fid_real_path ./datasets/fashion/train --distorated_path ./results/DPTN_fashion --gpu_id 0 --name DPTN_fashion_pil_bicubic --interpolation bicubic


#python eval.py --gt_path ./datasets/fashion/test --fid_real_path ./datasets/fashion/train --distorated_path ./results/c_to_c --gpu_id 0 --name c_to_c_cv2_bilinear --interpolation bilinear
#python eval.py --gt_path ./datasets/fashion/test --fid_real_path ./datasets/fashion/train --distorated_path ./results/c_to_c --gpu_id 0 --name c_to_c_cv2_nearest --interpolation nearest
#python eval.py --gt_path ./datasets/fashion/test --fid_real_path ./datasets/fashion/train --distorated_path ./results/c_to_c --gpu_id 0 --name c_to_c_cv2_bicubic --interpolation bicubic
#python eval.py --gt_path ./datasets/fashion/test --fid_real_path ./datasets/fashion/train --distorated_path ./results/c_to_c --gpu_id 0 --name c_to_c_cv2_area --interpolation area
#python eval.py --gt_path ./datasets/fashion/test --fid_real_path ./datasets/fashion/train --distorated_path ./results/c_to_c --gpu_id 0 --name c_to_c_pil_bilinear --interpolation bilinear
#python eval.py --gt_path ./datasets/fashion/test --fid_real_path ./datasets/fashion/train --distorated_path ./results/c_to_c --gpu_id 0 --name c_to_c_pil_nearest --interpolation nearest
#python eval.py --gt_path ./datasets/fashion/test --fid_real_path ./datasets/fashion/train --distorated_path ./results/c_to_c --gpu_id 0 --name c_to_c_pil_bicubic --interpolation bicubic

python eval.py --gt_path ./datasets/fashion/test --fid_real_path ./datasets/fashion/train --distorated_path ./results/DPTN_fashion --gpu_id 0 --name DPTN_fashion
python eval.py --gt_path ./datasets/fashion/test --fid_real_path ./datasets/fashion/train --distorated_path ./results/DPTN_2_stage --gpu_id 0 --name DPTN_2_stage
python eval.py --gt_path ./datasets/fashion/test --fid_real_path ./datasets/fashion/train --distorated_path ./results/DPTN_low_canonical --gpu_id 0 --name DPTN_low_canonical
python eval.py --gt_path ./datasets/fashion/test --fid_real_path ./datasets/fashion/train --distorated_path ./results/s_to_s --gpu_id 0 --name s_to_s
python eval.py --gt_path ./datasets/fashion/test --fid_real_path ./datasets/fashion/train --distorated_path ./results/c_to_s --gpu_id 0 --name c_to_s
python eval.py --gt_path ./datasets/fashion/test --fid_real_path ./datasets/fashion/train --distorated_path ./results/c_to_c --gpu_id 0 --name c_to_c
python eval.py --gt_path ./datasets/fashion/test --fid_real_path ./datasets/fashion/train --distorated_path ./results/c2c_spade_dec --gpu_id 0 --name c2c_spade_dec

