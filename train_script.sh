#python train.py --id baseline_canonical --tf_log --batchSize 26 --gpu_ids 3 --print_freq 1000
#python train.py --id base_spade_module --tf_log --gpu_ids 2 --print_freq 1000 --type_En_c spade --type_Dc spade --batchSize 12

# NIPA scripts
python train.py --id dptn_spade --tf_log --gpu_ids 1 --batchSize 10 --t_s_ratio 0.75 --type_En_c spade --type_Dc spade --continue_train



# red scripts, decoder full, port 6007
python train.py --id decoder_full --tf_log --gpu_ids 3 --type_En_c default --type_Dc spade --pose_nc 18 --batchSize 22
# red scripts, decoder img, port 6006
python train.py --id decoder_img --tf_log --gpu_ids 2 --type_En_c default --type_Dc spade --pose_nc 18 --batchSize 22

# NIPA scripts
python train.py --id spade --tf_log --gpu_ids 0 --batchSize 18 --type_En_c spade --type_Dc spade
python train.py --id baseline --tf_log --gpu_ids 1 --type_En_c default --type_Dc default --batchSize 20

# original red train
python train.py --id original --tf_log --gpu_ids 2 --type_En_c default --type_Dc default --pose_nc 18 --batchSize 22

# NIPA original train
python train.py --id c2c_spade_full --tf_log --gpu_ids 0 --type_En_c spade --type_Dc spade --batchSize 18
python train.py --id c2c_spade_dec --tf_log --gpu_ids 0 --type_En_c default --type_Dc spade --batchSize 18



# red original + limbmap train
python train.py --id original_fullmap --tf_log --gpu_ids 3 --type_En_c default --type_Dc default --pose_nc 37 --batchSize 22

# red original lsgan
python train.py --id original_lsgan --tf_log --gpu_ids 3 --type_En_c default --type_Dc default --pose_nc 18 --batchSize 22

# red original lsgan+model.train inference
python train.py --id original_vali --tf_log --gpu_ids 2 --type_En_c default --type_Dc default --pose_nc 18 --batchSize 22

# red original lsgan+instance norm
python train.py --id original_instance --tf_log --gpu_ids 1 --type_En_c default --type_Dc default --pose_nc 18 --batchSize 22

python train.py --id DPTN_higher --tf_log --gpu_ids 0 --type_En_c default --type_Dc default --pose_nc 18 --continue_train --batchSize 40

# DPTN_higher_original_SPADEFULL
python train.py --id DPTN_higher_spade_full --tf_log --gpu_ids 3 --type_En_c spade --type_Dc spade --pose_nc 18 --batchSize 14 --continue_train