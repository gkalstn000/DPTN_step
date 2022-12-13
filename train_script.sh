#python train.py --id baseline_canonical --tf_log --batchSize 26 --gpu_ids 3 --print_freq 1000
#python train.py --id base_spade_module --tf_log --gpu_ids 2 --print_freq 1000 --type_En_c spade --type_Dc spade --batchSize 12

# NIPA script
python train.py --id dptn_spade --tf_log --gpu_ids 1 --batchSize 10 --t_s_ratio 0.75 --type_En_c spade --type_Dc spade --continue_train



# red script, decoder full, port 6007
python train.py --id decoder_full --tf_log --gpu_ids 3 --type_En_c default --type_Dc spade --pose_nc 18 --batchSize 22
# red script, decoder img, port 6006
python train.py --id decoder_img --tf_log --gpu_ids 2 --type_En_c default --type_Dc spade --pose_nc 18 --batchSize 22

# NIPA script
python train.py --id spade --tf_log --gpu_ids 0 --batchSize 18 --type_En_c spade --type_Dc spade
python train.py --id baseline --tf_log --gpu_ids 1 --type_En_c default --type_Dc default --batchSize 20

# original red test
python train.py --id original --tf_log --gpu_ids 1 --type_En_c default --type_Dc default --pose_nc 18 --batchSize 5 --continue_train