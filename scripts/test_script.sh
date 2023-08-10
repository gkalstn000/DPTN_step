#python test.py --id DPTN_low_canonical --gpu_ids 2 --type_En_c default --type_Dc default --pose_nc 18  --batchSize 1
#python test.py --id c2c_spade_dec --gpu_ids 2 --type_En_c default --type_Dc spade  --batchSize 1


python test.py --id step_dptn --gpu_ids 0 --netG dptn --batchSize 30 --num_workers 10 --simple_test --which_epoch 0712 --save_id step_dptn_0712
python test.py --id step_dptn --gpu_ids 0 --netG dptn --batchSize 30 --num_workers 10 --simple_test --which_epoch 0801 --save_id step_dptn_0801
python test.py --id step_dptn --gpu_ids 0 --netG dptn --batchSize 30 --num_workers 10 --simple_test --which_epoch 0802 --save_id step_dptn_0802
python test.py --id step_dptn --gpu_ids 0 --netG dptn --batchSize 30 --num_workers 10 --simple_test --which_epoch 0803 --save_id step_dptn_0803
python test.py --id step_dptn --gpu_ids 0 --netG dptn --batchSize 30 --num_workers 10 --simple_test --which_epoch 0804 --save_id step_dptn_0804