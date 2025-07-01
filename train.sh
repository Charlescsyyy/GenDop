#!/usr/bin/env bash
set -x

PARTITION=3dobject_aigc_mmm
JOB_NAME=director3d
GPUS=1 # ${GPUS:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
CPUS_PER_TASK=${CPUS_PER_TASK:-1}
SRUN_ARGS=${SRUN_ARGS:-""}
# FILE_ID=$1 
# --quotatype=auto\
    # -w SH-IDC1-10-140-1-126 \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS} \
    -n1 -N1 \
    --mem-per-cpu=1000000 \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    --quotatype=reserved \
    --time=10-00:00:00 \
    accelerate launch --config_file acc_configs/gpu1.yaml main.py ArAE --workspace workspace_rebuttal --exp_name 'text-batchsize=16-discrete_bins=256-length=30-nofreeze-small-pure-ArtTraj' --cond_mode 'text' --text_key 'Movement' --num_cond_tokens 77 --discrete_bins 256 --pose_length 30 --batch_size 16 --hidden_dim 1024 --num_heads 8 --num_layers 12 --save_epoch 50

    # accelerate launch --config_file acc_configs/gpu1.yaml main.py ArAE --workspace workspace_ArtTraj --exp_name 'text-batchsize=16-discrete_bins=256-length=30-nofreeze-small-pure-ArtTraj-truescale' --cond_mode 'text' --text_key 'Movement' --num_cond_tokens 77 --discrete_bins 256 --pose_length 30 --batch_size 16 --hidden_dim 1024 --num_heads 8 --num_layers 12 --save_epoch 50

    # accelerate launch --config_file acc_configs/gpu1.yaml main.py ArAE --workspace workspace_ArtTraj --exp_name 'text-batchsize=16-discrete_bins=1024-length=30-nofreeze-small-mixed-ArtTraj' --cond_mode 'text' --text_key 'Concise Interaction' --num_cond_tokens 77 --discrete_bins 1024 --pose_length 30 --batch_size 16 --hidden_dim 1024 --num_heads 8 --num_layers 12 --save_epoch 50

    # accelerate launch --config_file acc_configs/gpu1.yaml main.py ArAE --workspace workspace_ArtTraj --exp_name 'depth+image+text-batchsize=16-discrete_bins=256-length=30-nofreeze-small-mixed-ArtTraj-resume' --resume "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/EdgeRunner/workspace_ArtTraj/depth+image+text-batchsize=16-discrete_bins=256-length=30-nofreeze-small-mixed-ArtTraj/best.safetensors" --cond_mode 'depth+image+text' --text_key 'Concise Interaction' --num_cond_tokens 591 --discrete_bins 256 --batch_size 16 --hidden_dim 1024 --num_heads 8 --num_layers 12 --save_epoch 50




    # accelerate launch --config_file acc_configs/gpu1.yaml main.py ArAE --workspace workspace_ArtTraj --exp_name 'depth+image+text-batchsize=32-discrete_bins=256-length=30-nofreeze-small-mixed-ArtTraj' --cond_mode 'depth+image+text' --text_key 'Concise Interaction' --num_cond_tokens 591 --discrete_bins 256 --batch_size 32 --hidden_dim 1024 --num_heads 8 --num_layers 12 --save_epoch 50


    # accelerate launch --config_file acc_configs/gpu1.yaml main.py ArAE --workspace workspace_ArtTraj --exp_name 'text-batchsize=16-discrete_bins=256-length=30-nofreeze-nonorm-small-pure-ArtTraj' --cond_mode 'text' --text_key 'Movement' --num_cond_tokens 77 --discrete_bins 256 --pose_length 30 --batch_size 16 --hidden_dim 1024 --num_heads 8 --num_layers 12 --save_epoch 50



    # accelerate launch --config_file acc_configs/gpu1.yaml main.py ArAE --workspace workspace_train_image --exp_name 'image+depth-batchsize=16-discrete_bins=256-nofreeze-realdepth-small-ArtTraj' --cond_mode 'image+depth' --num_cond_tokens 514 --discrete_bins 256 --batch_size 16 --hidden_dim 1024 --num_heads 8 --num_layers 12 --save_epoch 50
    # accelerate launch --config_file acc_configs/gpu1.yaml main.py ArAE --workspace workspace_train_image --exp_name 'image-batchsize=16-discrete_bins=256-nofreeze-realdepth-small-ArtTraj' --cond_mode 'image' --num_cond_tokens 257 --discrete_bins 256 --batch_size 16 --hidden_dim 1024 --num_heads 8 --num_layers 12 --save_epoch 50
    # accelerate launch --config_file acc_configs/gpu1.yaml main.py ArAE --workspace workspace_train_image --exp_name 'image-batchsize=16-discrete_bins=256-nofreeze-realdepth-ArtTraj' --cond_mode 'image' --num_cond_tokens 257 --discrete_bins 256 --batch_size 16 --save_epoch 50
    # accelerate launch --config_file acc_configs/gpu1.yaml main.py ArAE --workspace workspace_train_image --exp_name 'image+depth-batchsize=16-discrete_bins=256-nofreeze-realdepth-ArtTraj' --cond_mode 'image+depth' --num_cond_tokens 514 --discrete_bins 256 --batch_size 16 --save_epoch 50

    
    
    # accelerate launch --config_file acc_configs/gpu1.yaml main.py ArAE --workspace workspace_train_image --exp_name 'image+text-batchsize=16-discrete_bins=256-nofreeze-ArtTraj' --cond_mode 'image+text' --num_cond_tokens 334 --discrete_bins 256 --batch_size 16
    # accelerate launch --config_file acc_configs/gpu1.yaml main.py ArAE --workspace workspace_train_image --exp_name 'image+text-batchsize=16-discrete_bins=256-nofreeze-small-ArtTraj' --cond_mode 'image+text' --num_cond_tokens 334 --discrete_bins 256 --batch_size 16 --hidden_dim 1024 --num_heads 8 --num_layers 12
    # accelerate launch --config_file acc_configs/gpu1.yaml main.py ArAE --workspace workspace_train_image --exp_name 'depth+image+text-batchsize=16-discrete_bins=256-nofreeze-realdepth-ArtTraj' --cond_mode 'depth+image+text' --num_cond_tokens 591 --discrete_bins 256 --batch_size 16

    
    # --hidden_dim 1536 --num_heads 16 --num_layers 24
    #  --hidden_dim 512 --num_heads 8 --num_layers 8 
    # --hidden_dim 1024 --num_heads 8 --num_layers 12

    # accelerate launch --config_file acc_configs/gpu1.yaml main.py ArAE --workspace workspace_train_image --exp_name 'depth+image+text-batchsize=64-discrete_bins=256-lr=1e-5-nofreeze' --cond_mode 'depth+image+text' --num_cond_tokens 591 --batch_size 32 --lr 1e-5 --discrete_bins 256

    # accelerate launch --config_file acc_configs/gpu1.yaml main.py ArAE --workspace workspace_train_image --exp_name 'image+text-batchsize=64-discrete_bins=256-lr=1e-5-nofreeze' --cond_mode 'image+text' --num_cond_tokens 334 --batch_size 64 --lr 1e-5 --discrete_bins 256 




    
    #  --hidden_dim 1024 --num_heads 8 --num_layers 12

    # 
    
    # --resume /mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/EdgeRunner/workspace_train/text-batchsize=16-discrete_bins=512-freeze-tiny/ep0090/model.safetensors --start_epoch 91


    
    # --hidden_dim 512 --num_heads 8 --num_layers 8 
    

    # accelerate launch --config_file acc_configs/gpu1.yaml main.py ArAE --workspace workspace_train --exp_name 'image-batchsize=16-lr=1e-5-freeze' --cond_mode 'image' --num_cond_tokens 257 --batch_size 16 --lr 1e-5 
    
    # --hidden_dim 1024 --num_heads 8 --num_layers 12 

    # accelerate launch --config_file acc_configs/gpu1.yaml main.py ArAE --workspace workspace_train --exp_name 'image-batchsize=16-lr=1e-5' --cond_mode 'image' --num_cond_tokens 257 --freeze_encoder --batch_size 16 --lr 1e-5 --resume /mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/EdgeRunner/workspace_train/image-batchsize=16-lr=1e-5/ep0203/model.safetensors

# --hidden_dim 1536 --num_heads 16 --num_layers 24 --gradient_accumulation_steps 1 

    # accelerate launch --config_file acc_configs/gpu1.yaml main_dit.py DiT --workspace workspace_train_dit

    # accelerate launch --config_file acc_configs/gpu8.yaml main_dit.py DiT --workspace workspace_train_dit

