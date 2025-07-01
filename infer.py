'''
-----------------------------------------------------------------------------
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''

import os
import cv2
import tyro
import glob
import time
import json
import math
import shutil
import numpy as np
import torch
from PIL import Image
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import load_file

import kiui
import trimesh
from kiui.op import recenter

from core.options import AllConfigs, Options
from core.models import LMM
# from core.utils import load_mesh, normalize_mesh, get_tokenizer
from core.utils import monkey_patch_transformers
from core.utils import camera_to_token, camera_to_token_single, token_to_camera
from extrinsic2pyramid.util.camera_pose_visualizer import CameraPoseVisualizer
import matplotlib.pyplot as plt  # 导入 matplotlib 来调整图像大小

monkey_patch_transformers()

opt = tyro.cli(AllConfigs)

kiui.seed_everything(opt.seed)

# model
model = LMM(opt)

# resume pretrained checkpoint
if opt.resume is not None:
    if opt.resume.endswith('safetensors'):
        ckpt = load_file(opt.resume, device='cpu')
    else:
        ckpt = torch.load(opt.resume, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    print(f'[INFO] Loaded checkpoint from {opt.resume}')
else:
    print(f'[WARN] model randomly initialized, are you sane?')

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.half().eval().to(device)

# # tokenizer
# tokenizer, _ = get_tokenizer(opt)

# process function

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def matrix_to_quaternion(M: torch.Tensor) -> torch.Tensor:
    """
    Matrix-to-quaternion conversion method. Equation taken from 
    https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
    Args:
        M: rotation matrices, (... x 3 x 3)
    Returns:
        q: quaternion of shape (... x 4)
    """
    prefix_shape = M.shape[:-2]
    Ms = M.reshape(-1, 3, 3)

    trs = 1 + Ms[:, 0, 0] + Ms[:, 1, 1] + Ms[:, 2, 2]

    Qs = []

    for i in range(Ms.shape[0]):
        M = Ms[i]
        tr = trs[i]
        if tr > 0:
            r = torch.sqrt(tr) / 2.0
            x = ( M[ 2, 1] - M[ 1, 2] ) / ( 4 * r )
            y = ( M[ 0, 2] - M[ 2, 0] ) / ( 4 * r )
            z = ( M[ 1, 0] - M[ 0, 1] ) / ( 4 * r )
        elif ( M[ 0, 0] > M[ 1, 1]) and (M[ 0, 0] > M[ 2, 2]):
            S = torch.sqrt(1.0 + M[ 0, 0] - M[ 1, 1] - M[ 2, 2]) * 2 # S=4*qx 
            r = (M[ 2, 1] - M[ 1, 2]) / S
            x = 0.25 * S
            y = (M[ 0, 1] + M[ 1, 0]) / S 
            z = (M[ 0, 2] + M[ 2, 0]) / S 
        elif M[ 1, 1] > M[ 2, 2]: 
            S = torch.sqrt(1.0 + M[ 1, 1] - M[ 0, 0] - M[ 2, 2]) * 2 # S=4*qy
            r = (M[ 0, 2] - M[ 2, 0]) / S
            x = (M[ 0, 1] + M[ 1, 0]) / S
            y = 0.25 * S
            z = (M[ 1, 2] + M[ 2, 1]) / S
        else:
            S = torch.sqrt(1.0 + M[ 2, 2] - M[ 0, 0] -  M[ 1, 1]) * 2 # S=4*qz
            r = (M[ 1, 0] - M[ 0, 1]) / S
            x = (M[ 0, 2] + M[ 2, 0]) / S
            y = (M[ 1, 2] + M[ 2, 1]) / S
            z = 0.25 * S
        Q = torch.stack([r, x, y, z], dim=-1)
        Qs += [Q]

    return torch.stack(Qs, dim=0).reshape(*prefix_shape, 4)

def quaternion_slerp(
    q0, q1, fraction, spin: int = 0, shortestpath: bool = True
):
    """Return spherical linear interpolation between two quaternions.
    Args:
        quat0: first quaternion
        quat1: second quaternion
        fraction: how much to interpolate between quat0 vs quat1 (if 0, closer to quat0; if 1, closer to quat1)
        spin: how much of an additional spin to place on the interpolation
        shortestpath: whether to return the short or long path to rotation
    """
    d = (q0 * q1).sum(-1)
    if shortestpath:
        # invert rotation
        d[d < 0.0] = -d[d < 0.0]
        q1[d < 0.0] = q1[d < 0.0]

    d = d.clamp(0, 1.0)

    # theta = torch.arccos(d) * fraction
    # q2 = q1 - q0 * d
    # q2 = q2 / (q2.norm(dim=-1) + 1e-10)
    
    # return torch.cos(theta) * q0 + torch.sin(theta) * q2

    angle = torch.acos(d) + spin * math.pi
    isin = 1.0 / (torch.sin(angle)+ 1e-10)
    q0_ = q0 * torch.sin((1.0 - fraction) * angle) * isin
    q1_ = q1 * torch.sin(fraction * angle) * isin

    q = q0_ + q1_
    q[angle < 1e-5, :] = q0

    return q

def draw_json(c2ws, vis_path):
    
    output_dir = os.path.dirname(vis_path)
    parent_dir = os.path.dirname(output_dir)
    # vis_path = json_path.replace("_transforms", "_traj").replace(".json", ".png")
    # print(vis_path)
    # poses = []
    # data = json.load(open(json_path))['frames']
    # for frame in data:
    #     poses.append(frame['transform_matrix'])
        
    # poses = [poses[i] for i in range(0, len(poses), 2)]
    # c2ws = torch.tensor(poses)
    
    # ref_w2c = torch.inverse(c2ws[:1])
    # c2ws = ref_w2c.repeat(c2ws.shape[0], 1, 1) @ c2ws

    rangesize = torch.max(torch.abs(torch.tensor(c2ws[:, :3, 3]))) * 1.1

    # Prepare visualizer
    visualizer = CameraPoseVisualizer([-rangesize, rangesize], [-rangesize, rangesize], [-rangesize, rangesize])

    num_matrices = c2ws.shape[0]

    # Create a color gradient from red to purple
    colors = plt.cm.rainbow(np.linspace(1, 0, num_matrices))

    # Create three views
    views = [
        {'elev': 90, 'azim': -90, 'name': 'front'},
        {'elev': 180, 'azim': -90, 'name': 'top'},
        {'elev': 0, 'azim': 0, 'name': 'side'}
    ]
    
    image_paths = []

    for view in views:
        fig = plt.figure(figsize=(12, 12))  # Each image will be 4x12 inches
        visualizer = CameraPoseVisualizer([-rangesize, rangesize], [-rangesize, rangesize], [-rangesize, rangesize])

        for i in range(num_matrices):
            color = colors[i]
            # print(c2ws[i])
            visualizer.extrinsic2pyramid(c2ws[i], color, rangesize / 4)
        
        visualizer.ax.view_init(elev=view['elev'], azim=view['azim'])
        
        # Save each view as a separate image
        image_path = f"{parent_dir}/{view['name']}_view.png"
        os.makedirs(output_dir, exist_ok=True)
        visualizer.save(image_path)
        image_paths.append(image_path)
    
        
    # Now combine the three images horizontally
    images = [Image.open(img_path) for img_path in image_paths]
    images[-1] = images[-1].rotate(90, expand=True)

    # Resize images to fit the desired final size
    images = [img.crop((420, 420, 1980, 1980)) for img in images]
    images_resized = [img.resize((341, 341)) for img in images]

    # Concatenate images horizontally
    combined_image = np.concatenate([np.array(img) for img in images_resized], axis=1)

    # Save the final combined image
    final_image = Image.fromarray(combined_image)
    final_image.save(vis_path)

    print(f"Combined image saved at {vis_path}")
    


def sample_from_two_pose(pose_a, pose_b, fraction, noise_strengths=[0, 0]):
    """
    Args:
        pose_a: first pose
        pose_b: second pose
        fraction
    """
    def is_valid_rotation_matrix(matrix):
        should_be_identity = torch.matmul(matrix, matrix.transpose(-1, -2))
        identity = torch.eye(3, device=matrix.device).expand_as(should_be_identity)
        return torch.allclose(should_be_identity, identity, atol=1e-6) and torch.allclose(torch.det(matrix), torch.ones_like(matrix))

    quat_a = matrix_to_quaternion(pose_a[..., :3, :3])
    quat_b = matrix_to_quaternion(pose_b[..., :3, :3])
    dot = torch.sum(quat_a * quat_b, dim=-1, keepdim=True)
    quat_b = torch.where(dot < 0, -quat_b, quat_b)

    # quaternion = quaternion_slerp(quat_a, quat_b, fraction)

    cos_theta = torch.sum(quat_a * quat_b, dim=-1, keepdim=True)
    slerp_condition = cos_theta.abs() < 0.9995
    slerp_quat = quaternion_slerp(quat_a, quat_b, fraction)
    lerp_quat = (1 - fraction) * quat_a + fraction * quat_b
    lerp_quat = lerp_quat / lerp_quat.norm(dim=-1, keepdim=True)
    quaternion = torch.where(slerp_condition, slerp_quat, lerp_quat)
    
    quaternion = torch.nn.functional.normalize(quaternion + torch.randn_like(quaternion) * noise_strengths[0], dim=-1)

    R = quaternion_to_matrix(quaternion)
    T = (1 - fraction) * pose_a[..., :3, 3] + fraction * pose_b[..., :3, 3]
    T = T + torch.randn_like(T) * noise_strengths[1]

    new_pose = pose_a.clone()
    new_pose[..., :3, :3] = R
    new_pose[..., :3, 3] = T
    
    # assert is_valid_rotation_matrix(R), "Invalid rotation matrix"
    if not is_valid_rotation_matrix(R):
        print(new_pose.shape)
        print("Invalid rotation matrix")
        new_pose[..., :3, :3] = torch.eye(3)
        # new_pose[..., :3, 3] = T
        # return torch.eye(4)
        
    return new_pose

def sample_from_dense_cameras(dense_cameras, t, noise_strengths=[0, 0, 0, 0]):
    B, N, C = dense_cameras.shape
    B, M = t.shape
    
    left = torch.floor(t * (N-1)).long().clamp(0, N-2)
    right = left + 1
    fraction = t * (N-1) - left
    # print(fraction, right, left)
    # print("dense_cameras:", dense_cameras.device)
    # print("left:", left.device)
    a = torch.gather(dense_cameras, 1, left[..., None].repeat(1, 1, C))
    b = torch.gather(dense_cameras, 1, right[..., None].repeat(1, 1, C))

    new_pose = sample_from_two_pose(a[:, :, :12].reshape(B, M, 3, 4),
                                    b[:, :, :12].reshape(B, M, 3, 4), fraction, noise_strengths=noise_strengths[:2])

    new_ins = (1 - fraction) * a[:, :, 12:] + fraction * b[:, :, 12:]

    return torch.cat([new_pose.reshape(B, M, 12), new_ins], dim=2)  

def process_text(opt: Options, text, name):
    print(text)
    # name = text.split(' ')[0]
    os.makedirs(opt.workspace, exist_ok=True)

    assert opt.cond_mode == 'text'

    for i in range(opt.test_repeat):
        output_dir = os.path.join(opt.workspace, opt.resume.split('/')[-3]+'_'+opt.resume.split('/')[-2])
        output_path = os.path.join(opt.workspace, opt.resume.split('/')[-3]+'_'+opt.resume.split('/')[-2], f"{name}.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        t0 = time.time()

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                tokens = model.generate([text], max_new_tokens=opt.test_max_seq_length, clean=True)
        
        # single batch
        token = tokens[0]
        # print(token)
        coords = token[:-1].reshape(-1, 10)
        coords = torch.tensor(coords, dtype=torch.float32)
        # print(coords.shape)
        discrete_bins = opt.discrete_bins
        # 将 coords 恢复为原始的分成两部分
        coords_traj = coords[:, :7]
        coords_instri = coords[:, 7:]
        coords_scale = coords_instri[:, -1]

        # 恢复 coords_traj
        # 逆向公式： coords_traj -> (camera_tokens[:,:7] + 1) * 0.5 * discrete_bins
        temp_traj = coords_traj / (0.5 * discrete_bins) - 1

        # 恢复 coords_instri
        # 逆向公式： coords_instri -> (camera_tokens[:, 7:] / 10 * discrete_bins)
        temp_instri = coords_instri / (discrete_bins / 10)
        scale = torch.exp(coords_scale / discrete_bins * 4 - 2)

        # print(scale)

        # 拼接恢复的结果
        camera_tokens = torch.cat([temp_traj, temp_instri], dim=1)
        camera_tokens = camera_tokens.expand(1, -1, -1)
        # print(camera_tokens.shape)
        camera_pose = token_to_camera(camera_tokens, 512, 512)
        # print(camera_pose)
                 
        os.makedirs(os.path.join(output_dir), exist_ok=True)
        c2ws = np.array(camera_pose[:, :, :12].cpu())
        
        scale_value = np.array(scale[0].cpu())
        # print(scale_value)
        c2ws = c2ws.reshape((-1, 3, 4))
        c2ws[:, :3, 3] = c2ws[:, :3, 3] * scale_value
        row_to_add = np.array([0, 0, 0, 1])
        c2ws = np.array([np.vstack((matrix, row_to_add)) for matrix in c2ws])
        
        draw_json(c2ws, os.path.join(output_dir, f"{name}_traj.png"))
        # draw_json(c2ws, os.path.join(output_dir, f"{name}_traj.png"))
        
        def pose_normalize(camera_pose, pred_pose_path):
            camera_pose = camera_pose
            # print(camera_pose.shape)
            transforms_path = pred_pose_path

            f_x, f_y, c_x, c_y, w, h = camera_pose[0][0][-6:].tolist()
            # Create a dictionary of intrinsic parameters
            transforms_dict = {
                "w": w,
                "h": h,
                "fl_x": f_x,  # Focal length in x direction
                "fl_y": f_y,  # Focal length in y direction
                "cx": c_x,    # Principal point in x
                "cy": c_y,     # Principal point in y
                'frames': []
            }
            # print(transforms_dict)
            traj_tensor = camera_pose[:,:,:12]
            # print(traj_tensor)
            camera_list = []
            for i in range(120):
                t = torch.full((1, 1), fill_value=i/120)
                camera = sample_from_dense_cameras(traj_tensor, t)
                camera_list.append(camera[0])
            camera_tensor = torch.cat(camera_list, dim=0)  # Concatenate along the batch dimension (dim=0)
            # print(camera_tensor.shape)
            # print(camera_tensor.view(1, 120, 12))
            camera_numpy = camera_tensor.clone().cpu().numpy()
            # transform_matrixs = []
            for idx, row in enumerate(camera_numpy):
                RT = row.reshape(3, 4)
                transform_matrix = np.vstack([RT, [0, 0, 0, 1]])
                transform_matrix_list = transform_matrix.tolist()
                # Prepare frame data
                frame_data = {
                    "transform_matrix": transform_matrix_list,
                    "monst3r_im_id": idx + 1  # Assuming colmap_im_id is an index starting from 1
                }
                transforms_dict['frames'].append(frame_data)
                # transform_matrixs.append(transform_matrix)
                
            # print(np.array(transform_matrixs))
            # for matrix in transform_matrixs:
            #     print("    ", matrix, ",")
            # print("])")
            
            # Save the transforms dictionary to a JSON file
            with open(transforms_path, 'w') as f:
                json.dump(transforms_dict, f, indent=4)
                
        def save_results(output_dir, name, camera_pose):
            pred_pose_path = os.path.join(output_dir, f"{name}_transforms_pred.json")
            pose_normalize(camera_pose, pred_pose_path)
            
        save_results(output_dir, name, camera_pose)
        
        # # # 计算范围大小，使用numpy相关函数操作
        # # rangesize = torch.max(torch.abs(torch.tensor(c2ws[:, :3, 3])))*1.1


        # # fig = plt.figure(figsize=(12, 12))  # 宽 12 英寸，高 8 英寸
        # # visualizer = CameraPoseVisualizer([-rangesize, rangesize], [-rangesize, rangesize], [-rangesize, rangesize])
        # # num_matrices = c2ws.shape[0]
        # # colors = plt.cm.rainbow(np.linspace(1, 0, num_matrices))
        # # for i in range(num_matrices):
        # #     color = colors[i]  # 获取渐变中的颜色
        # #     # print(color)
        # #     visualizer.extrinsic2pyramid(c2ws[i], color, rangesize/4)
        # # visualizer.ax.view_init(elev=135, azim=-90)  # 调整视角的仰角和方位角
        # # visualizer.save(os.path.join(output_dir, 'pose', f"{'_'.join(text[:200].split())}_{i}.png"))

        # json_path = os.path.join(output_dir, f"{'_'.join(text[:200].split())}_{i}.json")
        # entry = {
        #     'text': text,
        #     'pose': camera_pose.tolist()  # 将 tensor 转换为列表
        # }
        # json.dump(entry, open(json_path, 'w'))
        
        # # json.dump(camera_pose.numpy().tolist(), open(output_path, 'w'), indent=4)
        
        # # timing
        # torch.cuda.synchronize()
        # t1 = time.time()
        # print(f'[INFO] Processing, time = {t1 - t0:.4f}s')
    
# process function
def process_data(opt, image_path, text_path=None, depth_path=None):
    """
    处理图像、文本和深度数据的统一函数。
    :param opt: 配置参数对象。
    :param image_path: 图像文件路径。
    :param text_path: 文本文件路径（可选）。
    :param depth_path: 深度文件路径（可选）。
    """
    # 提取名称
    name = 'test/' + image_path.split('/')[-2] + '/' + image_path.split('/')[-1][:-8]
    print(name)
    
    output_dir = os.path.join(opt.workspace, opt.resume.split('/')[-3] + '_' + opt.resume.split('/')[-2])
    os.makedirs(output_dir, exist_ok=True)
    
    new_traj_path = os.path.join(output_dir, f"{name}_transforms_pred.json")
    if os.path.exists(new_traj_path):
        print(f"Skipping {name} as it already exists.")
        return

    # 读取文本（如果存在）
    text = None
    if text_path is not None:
        info = json.load(open(text_path, 'r'))
        text = info[opt.text_key]
        print(text)
        # text = info['Concise Interaction']

    # 检查条件模式是否匹配
    if opt.cond_mode == 'text':
        assert text_path is not None, "text_path is required for 'text' mode"
    elif opt.cond_mode == 'image+text':
        assert text_path is not None, "text_path is required for 'image+text' mode"
    elif opt.cond_mode == 'depth+image+text':
        assert text_path is not None and depth_path is not None, "text_path and depth_path are required for 'depth+image+text' mode"
    elif opt.cond_mode == 'image':
        assert text_path is None and depth_path is None, "text_path and depth_path should be None for 'image' mode"
    elif opt.cond_mode == 'image+depth':
        assert depth_path is not None, "depth_path is required for 'image+depth' mode"
    else:
        raise ValueError(f"Unsupported cond_mode: {opt.cond_mode}")

    # 标准化图像
    def standard_image(rgb_path, target_height=512, target_width=512):
        image = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0  # [H, W, 4]
        image = image[..., [2, 1, 0]]  # BGR to RGB
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()  # [C, H, W]
        height, width = image_tensor.shape[1], image_tensor.shape[2]

        if height > target_height:
            start_y = (height - target_height) // 2
            image_tensor = image_tensor[:, start_y:start_y + target_height, :]
        
        if width > target_width:
            start_x = (width - target_width) // 2
            image_tensor = image_tensor[:, :, start_x:start_x + target_width]

        # 如果图像尺寸小于目标尺寸，进行填充
        if image_tensor.shape[1] < target_height or image_tensor.shape[2] < target_width:
            padded_image = torch.zeros((3, target_height, target_width), dtype=torch.float32)
            
            # 计算需要填充的空白区域
            top_padding = (target_height - image_tensor.shape[1]) // 2
            bottom_padding = target_height - image_tensor.shape[1] - top_padding
            left_padding = (target_width - image_tensor.shape[2]) // 2
            right_padding = target_width - image_tensor.shape[2] - left_padding

            # 将原图像放置到填充后的背景上
            padded_image[:, top_padding:top_padding + image_tensor.shape[1], left_padding:left_padding + image_tensor.shape[2]] = image_tensor
            image_tensor = padded_image
        return image_tensor

    def standard_depth(depth_path, target_height=512, target_width=512):
        # 读取深度图，假设深度图是一个单通道的浮动数组（通常深度图是浮点数）
        depth_image = np.load(depth_path).astype(np.float32)  # [H, W]
        depth_tensor = torch.from_numpy(depth_image).unsqueeze(0).float()  # [0, 1]
        
        # # 严格归一化到 [0, 1] 范围
        # min_depth = torch.min(depth_tensor)
        # max_depth = torch.max(depth_tensor)
        # depth_tensor = (depth_tensor - min_depth) / (max_depth - min_depth)
        
        height, width = depth_tensor.shape[1], depth_tensor.shape[2]

        # 如果图像高度大于目标高度，进行裁剪
        if height > target_height:
            start_y = (height - target_height) // 2
            depth_tensor = depth_tensor[:, start_y:start_y + target_height, :]

        # 如果图像宽度大于目标宽度，进行裁剪
        if width > target_width:
            start_x = (width - target_width) // 2
            depth_tensor = depth_tensor[:, :, start_x:start_x + target_width]

        # 如果图像尺寸小于目标尺寸，进行填充
        if depth_tensor.shape[1] < target_height or depth_tensor.shape[2] < target_width:
            padded_depth = torch.zeros((1, target_height, target_width), dtype=torch.float32)
            
            # 计算需要填充的空白区域
            top_padding = (target_height - depth_tensor.shape[1]) // 2
            bottom_padding = target_height - depth_tensor.shape[1] - top_padding
            left_padding = (target_width - depth_tensor.shape[2]) // 2
            right_padding = target_width - depth_tensor.shape[2] - left_padding

            # 将原深度图放置到填充后的背景上
            padded_depth[:, top_padding:top_padding + depth_tensor.shape[1], left_padding:left_padding + depth_tensor.shape[2]] = depth_tensor
            depth_tensor = padded_depth

        return depth_tensor

    # 处理图像
    if image_path is not None and opt.cond_mode != 'text':
        rgb = standard_image(image_path, target_height=opt.target_height, target_width=opt.target_width).to(device)
        rgb_show = rgb.permute(1, 2, 0)  # [3, H, W] -> [H, W, 3]
        rgb_batch = rgb.expand(1, -1, -1, -1)
        kiui.write_image(os.path.join(output_dir, f"{name}_rgb.png"), rgb_show)

    # 处理深度图（如果存在）
    depth_batch = None
    if depth_path is not None:
        depth = standard_depth(depth_path, target_height=opt.target_height, target_width=opt.target_width)
        print(depth.shape)
        depth_show = depth.squeeze()
        plt.figure(figsize=(12, 12))
        sns.heatmap(depth_show, cmap='viridis')
        plt.savefig(os.path.join(output_dir, f"{name}_depth.png"))
        depth_batch = depth.expand(1, -1, -1, -1)

    # 准备输入条件
    if opt.cond_mode == 'text':
        conds = [text]
    elif opt.cond_mode == 'image':
        conds = rgb_batch
    elif opt.cond_mode == 'image+text':
        conds = [[text], rgb_batch]
    elif opt.cond_mode == 'image+depth':
        conds = [depth_batch, rgb_batch]
    elif opt.cond_mode == 'depth+image+text':
        conds = [[text], rgb_batch, depth_batch]
    else:
        raise ValueError(f"Unsupported cond_mode: {opt.cond_mode}")

    # 生成结果
    for i in range(opt.test_repeat):
        t0 = time.time()

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                tokens = model.generate(conds, max_new_tokens=opt.test_max_seq_length, clean=True)
        t1 = time.time()
        print(f'[INFO] Processing, time = {t1 - t0:.4f}s')
        # 处理生成的 tokens
        token = tokens[0]
        # print(token)
        # print(token[:-1].shape[0])
        if token[:-1].shape[0] != opt.pose_length * 10:
            token = torch.tensor([256, 128, 128, 128, 128, 128, 128, 36, 64, 60]) / 256 * opt.discrete_bins
            token = token.repeat(opt.pose_length)
            coords = token.reshape(-1, 10)
        else:
            coords = token[:-1].reshape(-1, 10)
        coords = torch.tensor(coords, dtype=torch.float32)
        discrete_bins = opt.discrete_bins

        coords_traj = coords[:, :7]
        coords_instri = coords[:, 7:]
        coords_scale = coords_instri[:, -1]

        temp_traj = coords_traj / (0.5 * discrete_bins) - 1
        temp_instri = coords_instri / (discrete_bins / 10)
        scale = torch.exp(coords_scale / discrete_bins * 4 - 2)

        camera_tokens = torch.cat([temp_traj, temp_instri], dim=1)
        camera_tokens = camera_tokens.expand(1, -1, -1)
        camera_pose = token_to_camera(camera_tokens, 512, 512)
        
        c2ws = np.array(camera_pose[:, :, :12].cpu())
        scale_value = np.array(scale[0].cpu())
        c2ws = c2ws.reshape((-1, 3, 4))
        c2ws[:, :3, 3] = c2ws[:, :3, 3] * scale_value

        row_to_add = np.array([0, 0, 0, 1])
        c2ws = np.array([np.vstack((matrix, row_to_add)) for matrix in c2ws])
        
        # entry = {
        #     'text': text if text is not None else '',
        #     'pose': camera_pose.tolist()
        # }
        
        def pose_normalize(camera_pose, pred_pose_path):
            camera_pose = camera_pose
            # print(camera_pose.shape)
            transforms_path = pred_pose_path

            f_x, f_y, c_x, c_y, w, h = camera_pose[0][0][-6:].tolist()
            # Create a dictionary of intrinsic parameters
            transforms_dict = {
                "w": w,
                "h": h,
                "fl_x": f_x,  # Focal length in x direction
                "fl_y": f_y,  # Focal length in y direction
                "cx": c_x,    # Principal point in x
                "cy": c_y,     # Principal point in y
                'frames': []
            }
            # print(transforms_dict)
            traj_tensor = camera_pose[:,:,:12]
            # print(traj_tensor)
            camera_list = []
            for i in range(120):
                t = torch.full((1, 1), fill_value=i/120)
                camera = sample_from_dense_cameras(traj_tensor, t)
                camera_list.append(camera[0])
            camera_tensor = torch.cat(camera_list, dim=0)  # Concatenate along the batch dimension (dim=0)
            # print(camera_tensor.shape)
            # print(camera_tensor.view(1, 120, 12))
            camera_numpy = camera_tensor.clone().cpu().numpy()
            # transform_matrixs = []
            for idx, row in enumerate(camera_numpy):
                RT = row.reshape(3, 4)
                transform_matrix = np.vstack([RT, [0, 0, 0, 1]])
                transform_matrix_list = transform_matrix.tolist()
                # Prepare frame data
                frame_data = {
                    "transform_matrix": transform_matrix_list,
                    "monst3r_im_id": idx + 1  # Assuming colmap_im_id is an index starting from 1
                }
                transforms_dict['frames'].append(frame_data)
                # transform_matrixs.append(transform_matrix)
                
            # print(np.array(transform_matrixs))
            # for matrix in transform_matrixs:
            #     print("    ", matrix, ",")
            # print("])")
            
            # Save the transforms dictionary to a JSON file
            with open(transforms_path, 'w') as f:
                json.dump(transforms_dict, f, indent=4)
                
        def save_results(output_dir, name, camera_pose):
            gt_caption_path = image_path.replace("_rgb.png", "_caption.json")
            new_caption_path = os.path.join(output_dir, f"{name}_caption.json")
            os.makedirs(os.path.dirname(new_caption_path), exist_ok=True)
            shutil.copy(gt_caption_path, new_caption_path)
            
            gt_pose_path = image_path.replace("_rgb.png", "_transforms_cleaning.json")
            new_pose_path = os.path.join(output_dir, f"{name}_transforms_ref.json")
            shutil.copy(gt_pose_path, new_pose_path)
            
            pred_pose_path = os.path.join(output_dir, f"{name}_transforms_pred.json")
            pose_normalize(camera_pose, pred_pose_path)
        
        
        assert i == 0
        
        # if int(name.split('/')[-2].split('_')[1]) % 100 == 0 or int(name.split('/')[-2].split('_')[0]) == 0:
        draw_json(c2ws, os.path.join(output_dir, f"{name}_traj.png"))
        gt_traj_path = image_path.replace("_rgb.png", "_traj_cleaning.png")
        new_traj_path = os.path.join(output_dir, f"{name}_traj_GT.png")
        shutil.copy(gt_traj_path, new_traj_path)
            
        save_results(output_dir, name, camera_pose)
            
            # if i == 0:
            #     # json.dump(entry, open(os.path.join(output_dir, f"{name}_info.json"), 'w'), indent=4)
            # else:
            #     draw_json(c2ws, os.path.join(output_dir, f"{name}_{i}_traj.png"))
            #     save_results(output_dir, name, camera_pose)
            #     # json.dump(entry, open(os.path.join(output_dir, f"{name}_{i}_info.json"), 'w'), indent=4)

            # 列出GT traj

        
        # 计时
        torch.cuda.synchronize()
        
assert opt.test_path is not None
        
if opt.cond_mode == 'text': 
    print("Start processing text")
    if os.path.isdir(opt.test_path):
        image_paths = glob.glob(os.path.join(opt.test_path, "*_rgb.png"))
        print("Number of images:", len(image_paths))
        for image_path in sorted(image_paths):
            text_path = image_path.replace("_rgb.png", "_caption.json")
            process_data(opt, image_path, text_path)
elif opt.cond_mode == 'image': 
    print("Start processing image")
    if os.path.isdir(opt.test_path):
        image_paths = glob.glob(os.path.join(opt.test_path, "*/*_rgb.png"))
        print("Number of images:", len(image_paths))
        for image_path in image_paths:
            text_path = image_path.replace("_rgb.png", "_caption.json")
            process_data(opt, image_path)
elif opt.cond_mode == 'image+text': 
    print("Start processing image+text")
    if os.path.isdir(opt.test_path):
        image_paths = glob.glob(os.path.join(opt.test_path, "*/*_rgb.png"))
        print("Number of images:", len(image_paths))
        for image_path in image_paths:
            text_path = image_path.replace("_rgb.png", "_caption.json")
            process_data(opt, image_path, text_path)
elif opt.cond_mode == 'image+depth': 
    print("Start processing image+text")
    if os.path.isdir(opt.test_path):
        image_paths = glob.glob(os.path.join(opt.test_path, "*/*_rgb.png"))
        print("Number of images:", len(image_paths))
        for image_path in image_paths[::100]:
            # text_path = image_path.replace("_rgb.png", "_caption.json")
            depth_path = image_path.replace("_rgb.png", "_depth.npy")
            process_data(opt, image_path, None, depth_path)
elif opt.cond_mode == 'depth+image+text': 
    print("Start processing depth+image+text")
    if os.path.isdir(opt.test_path):
        image_paths = glob.glob(os.path.join(opt.test_path, "*/*_rgb.png"))
        print("Number of images:", len(image_paths))
        for image_path in image_paths:
            print(image_path)
            text_path = image_path.replace("_rgb.png", "_caption.json")
            depth_path = image_path.replace("_rgb.png", "_depth.npy")
            process_data(opt, image_path, text_path, depth_path)
print(os.path.join(opt.workspace, opt.resume.split('/')[-3]+'_'+opt.resume.split('/')[-2]))