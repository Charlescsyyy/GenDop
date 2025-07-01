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
    --quotatype=auto \
    --time=1-00:00:00 \
    python infer.py ArAE --workspace Results_image --resume "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/EdgeRunner/workspace_ArtTraj/depth+image+text-batchsize=16-discrete_bins=256-length=30-nofreeze-small-mixed-ArtTraj-resume/ep0050/model.safetensors" --test_path "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/DATA/ShotTraj/dataset/ArtTraj/test" --cond_mode 'depth+image+text' --text_key 'Concise Interaction' --num_cond_tokens 591 --discrete_bins 256 --pose_length 30 --hidden_dim 1024 --num_heads 8 --num_layers 12
    
    
    # python infer.py ArAE --workspace workspace_infer_image --resume "workspace_train_image/depth+image+text-batchsize=16-discrete_bins=256-nofreeze-realdepth-small-ArtTraj/best.safetensors" --test_path "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/DATA/ShotTraj/dataset/ArtTraj/test" --cond_mode 'depth+image+text' --num_cond_tokens 591 --discrete_bins 256 --hidden_dim 1024 --num_heads 8 --num_layers 12
    # python infer.py ArAE --workspace workspace_infer_image --resume "workspace_train_image/depth+image+text-batchsize=16-discrete_bins=256-nofreeze-realdepth-ArtTraj/best.safetensors" --test_path "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/DATA/ShotTraj/dataset/ArtTraj/test" --cond_mode 'depth+image+text' --num_cond_tokens 591 --discrete_bins 256
    # python infer.py ArAE --workspace workspace_infer_image --resume "workspace_train_image/image+text-batchsize=16-discrete_bins=256-nofreeze-small-ArtTraj/best.safetensors" --test_path "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/DATA/ShotTraj/dataset/ArtTraj/test" --cond_mode 'image+text' --num_cond_tokens 334 --discrete_bins 256 --hidden_dim 1024 --num_heads 8 --num_layers 12
    # python infer.py ArAE --workspace workspace_infer_image --resume "workspace_train_image/image+text-batchsize=16-discrete_bins=256-nofreeze-ArtTraj/best.safetensors" --test_path "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/DATA/ShotTraj/dataset/ArtTraj/test" --cond_mode 'image+text' --num_cond_tokens 334 --discrete_bins 256
    
    # python infer.py ArAE --workspace workspace_infer_image --resume "workspace_train_image/image+text-batchsize=16-discrete_bins=256-nofreeze-small-ArtTraj/best.safetensors" --test_path "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/DATA/ShotTraj/dataset/ArtTraj/test" --cond_mode 'image+text' --num_cond_tokens 334 --discrete_bins 256 --hidden_dim 1024 --num_heads 8 --num_layers 12
    # python infer.py ArAE --workspace workspace_infer_image --resume "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/EdgeRunner/workspace_train_image/depth+image+text-batchsize=16-discrete_bins=256-nofreeze-normdepth-small-ArtTraj/best.safetensors" --test_path "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/DATA/ShotTraj/dataset/ArtTraj/test" --cond_mode 'depth+image+text' --num_cond_tokens 591 --discrete_bins 256 --hidden_dim 1024 --num_heads 8 --num_layers 12

    # python infer.py ArAE --workspace workspace_infer --resume "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/EdgeRunner/workspace_train/image+text-batchsize=16-discrete_bins=128-lr=1e-5-nofreeze/ep0030/model.safetensors" --test_path "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/EdgeRunner/assets/images" --cond_mode 'image+text' --num_cond_tokens 334 --discrete_bins 128

    
    # --hidden_dim 1024 --num_heads 8 --num_layers 12




    # python infer.py ArAE --workspace workspace_infer --resume "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/EdgeRunner/workspace_train/image-batchsize=16-lr=1e-5-nofreeze/ep0100/model.safetensors" --test_path "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/EdgeRunner/assets/images" --cond_mode 'image' --num_cond_tokens 257






    # python infer.py ArAE --workspace workspace_infer --resume "//mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/EdgeRunner/workspace_train/text-batchsize=64-lr=1e-4/best.safetensors" --test_path "/mnt/petrelfs/zhangmengchen/20241011_CameraTrajectory/EdgeRunner/assets/texts/text_file.txt" --cond_mode 'text'

