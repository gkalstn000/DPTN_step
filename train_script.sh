#python train.py --id baseline_canonical --tf_log --batchSize 26 --gpu_ids 3 --print_freq 1000
#python train.py --id base_spade_module --tf_log --gpu_ids 2 --print_freq 1000 --type_En_c spade --type_Dc spade --batchSize 12

# NIPA script
python train.py --id dptn_spade --tf_log --gpu_ids 0,1 --batchSize 20 --t_s_ratio 0.75 --type_En_c spade --type_Dc spade --continue_train