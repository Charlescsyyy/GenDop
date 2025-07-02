#!/usr/bin/env bash
set -x

PARTITION=3dobject_aigc_mmm_t
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
    accelerate launch --config_file acc_configs/gpu1.yaml main.py ArAE --workspace workspace --exp_name 'text_rgbd' --cond_mode 'depth+image+text' --text_key 'Concise Interaction' --num_cond_tokens 591 --discrete_bins 256 --pose_length 30 --hidden_dim 1024 --num_heads 8 --num_layers 12 --save_epoch 50

    # accelerate launch --config_file acc_configs/gpu1.yaml main.py ArAE --workspace workspace --exp_name 'text_motion' --cond_mode 'text' --text_key 'Movement' --num_cond_tokens 77 --discrete_bins 256 --pose_length 30 --hidden_dim 1024 --num_heads 8 --num_layers 12 --save_epoch 50
    # accelerate launch --config_file acc_configs/gpu1.yaml main.py ArAE --workspace workspace --exp_name 'text_directorial' --cond_mode 'text' --text_key 'Concise Interaction' --num_cond_tokens 77 --discrete_bins 256 --pose_length 30 --hidden_dim 1024 --num_heads 8 --num_layers 12 --save_epoch 50