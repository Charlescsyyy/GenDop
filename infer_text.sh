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
    --quotatype=auto \
    --time=0-10:00:00 \
    pip install flash-attn --no-build-isolation

    # python infer.py ArAE --workspace outputs --resume "checkpoints/text_motion.safetensors" --test_path "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/DATA/ShotTraj/dataset/ArtTraj/test" --cond_mode 'text' --num_cond_tokens 77 --discrete_bins 256 --hidden_dim 1024 --num_heads 8 --num_layers 12 
    
    # python infer.py ArAE --workspace Cases --resume "checkpoints/text_directorial.safetensors" --test_path "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/DATA/ShotTraj/dataset/ArtTraj/test" --cond_mode 'text' --num_cond_tokens 77 --discrete_bins 256 --hidden_dim 1024 --num_heads 8 --num_layers 12 

    # python infer.py ArAE --workspace Ego3d --resume "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/EdgeRunner/workspace_ArtTraj/text-batchsize=16-discrete_bins=256-length=30-nofreeze-small-pure-ArtTraj/ep0100/model.safetensors" --test_path "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/DataDoP/gendop_ego4d/Dataset" --cond_mode 'text' --text_key 'Concise Interaction' --num_cond_tokens 77 --discrete_bins 256 --pose_length 30 --batch_size 16 --hidden_dim 1024 --num_heads 8 --num_layers 12
    # python infer.py ArAE --workspace Cases --resume "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/EdgeRunner/workspace_ArtTraj/text-batchsize=16-discrete_bins=256-length=30-nofreeze-small-pure-ArtTraj/ep0100/model.safetensors" --test_path "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/EdgeRunner/assets/texts/text_v5.txt" --cond_mode 'text' --text_key 'Movement' --num_cond_tokens 77 --discrete_bins 256 --pose_length 30 --batch_size 16 --hidden_dim 1024 --num_heads 8 --num_layers 12
    # python infer.py ArAE --workspace Results_epoch100 --resume "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/EdgeRunner/workspace_ArtTraj/text-batchsize=16-discrete_bins=256-length=30-nofreeze-small-mixed-ArtTraj/ep0100/model.safetensors" --test_path "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/DATA/ShotTraj/dataset/ArtTraj/test" --cond_mode 'text' --text_key 'Concise Interaction' --num_cond_tokens 77 --discrete_bins 256 --pose_length 30 --batch_size 16 --hidden_dim 1024 --num_heads 8 --num_layers 12
    # python infer.py ArAE --workspace Results --resume "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/EdgeRunner/workspace_ArtTraj/text-batchsize=16-discrete_bins=256-length=120-nofreeze-small-pure-ArtTraj/ep0200/model.safetensors" --test_path "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/DATA/ShotTraj/dataset/ArtTraj/test" --cond_mode 'text' --text_key 'Movement' --num_cond_tokens 77 --discrete_bins 256 --pose_length 120 --batch_size 16 --hidden_dim 1024 --num_heads 8 --num_layers 12
    # python infer.py ArAE --workspace Results --resume "workspace_ArtTraj/text-batchsize=16-discrete_bins=256-length=30-nofreeze-small-mixed-ArtTraj/ep0150/model.safetensors" --test_path "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/DATA/ShotTraj/dataset/ArtTraj/test" --cond_mode 'text' --text_key 'Concise Interaction' --num_cond_tokens 77 --discrete_bins 256 --pose_length 30 --batch_size 16 --hidden_dim 1024 --num_heads 8 --num_layers 12
    # python infer.py ArAE --workspace workspace_infer --resume "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/EdgeRunner/workspace_train_text/text-batchsize=64-discrete_bins=256-nofreeze-small/ep0999/model.safetensors" --test_path "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/director3d/assets/newdata.txt" --cond_mode 'text' --num_cond_tokens 77 --discrete_bins 256 --hidden_dim 1024 --num_heads 8 --num_layers 12 
    # --hidden_dim 1024 --num_heads 8 --num_layers 12 --pose_length 15
    # python infer.py ArAE --workspace workspace_infer --resume "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/EdgeRunner/workspace_train_text/text-batchsize=256-discrete_bins=256-nofreeze/best.safetensors" --test_path "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/EdgeRunner/assets/texts/text_file.txt" --cond_mode 'text' --num_cond_tokens 77 --discrete_bins 256
    # python infer.py ArAE --workspace workspace_infer --resume "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/EdgeRunner/workspace_train_text/text-batchsize=64-discrete_bins=256-nofreeze/best.safetensors" --test_path "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/EdgeRunner/assets/texts/text_file.txt" --cond_mode 'text' --num_cond_tokens 77 --discrete_bins 256
    # python infer.py ArAE --workspace workspace_infer --resume "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/EdgeRunner/workspace_train_text/text-batchsize=64-discrete_bins=256-pose_length=15-nofreeze-small/best.safetensors" --test_path "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/EdgeRunner/assets/texts/text_file.txt" --cond_mode 'text' --num_cond_tokens 77 --discrete_bins 256 --hidden_dim 1024 --num_heads 8 --num_layers 12 --pose_length 15
    # python infer.py ArAE --workspace workspace_infer --resume "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/EdgeRunner/workspace_train_text/text-batchsize=256-discrete_bins=256-nofreeze-small/best.safetensors" --test_path "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/EdgeRunner/assets/texts/text_file.txt" --cond_mode 'text' --num_cond_tokens 77 --discrete_bins 256 --hidden_dim 1024 --num_heads 8 --num_layers 12
    
    
    # --pose_length 15
    # --hidden_dim 1536 --num_heads 16 --num_layers 24
    # --hidden_dim 1024 --num_heads 8 --num_layers 12 
    # --hidden_dim 512 --num_heads 8 --num_layers 8 
    # python infer.py ArAE --workspace workspace_infer --resume "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/EdgeRunner/workspace_train/text-batchsize=16-lr=1e-5-freeze-small/best.safetensors" --test_path "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/EdgeRunner/assets/texts/text_file.txt" --cond_mode 'text' --num_cond_tokens 77 --hidden_dim 1024 --num_heads 8 --num_layers 12 

    # python infer.py ArAE --workspace workspace_infer --resume "//mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/EdgeRunner/workspace_train/text-batchsize=16-lr=1e-5-nofreeze/best.safetensors" --test_path "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/EdgeRunner/assets/texts/text_file.txt" --cond_mode 'text' --num_cond_tokens 77 --hidden_dim 1024 --num_heads 8 --num_layers 12 
    # python infer.py ArAE --workspace workspace_infer --resume "//mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/EdgeRunner/workspace_train/text-batchsize=16-lr=1e-5-nofreeze/best.safetensors" --test_path "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/EdgeRunner/assets/texts/text_file.txt" --cond_mode 'text' --num_cond_tokens 77

    # python infer.py ArAE --workspace workspace_infer --resume "//mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/EdgeRunner/workspace_train/text-batchsize=16-lr=1e-5-nofreeze/best.safetensors" --test_path "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/EdgeRunner/assets/images" --cond_mode 'image' --num_cond_tokens 257
# python infer.py ArAE --workspace workspace_infer --resume "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/EdgeRunner/workspace_train_text/text-batchsize=256-discrete_bins=256-nofreeze-small/ep0050/model.safetensors" --test_path "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/EdgeRunner/assets/text_file.txt" --cond_mode 'text'

