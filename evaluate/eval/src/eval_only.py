import torch
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List
from torchtyping import TensorType
from src.metrics.modules.caption import CaptionMetrics
from src.metrics.modules.fcd import FrechetCLaTrDistance
from src.metrics.modules.prdc import ManifoldMetrics
from src.metrics.modules.clatr_score import CLaTrScore
from utils.random_utils import set_random_seed

# # ------------------------------------------------------------------------------------- #
# # 输入数据的类型声明
# # ------------------------------------------------------------------------------------- #
Traj = TensorType["num_samples", "num_cams", 4, 4]  # 摄像机轨迹
Feat = TensorType["num_samples", "num_feats"]  # 特征
Mask = TensorType["num_samples", "num_cams"]  # 掩码（用于文本描述）
# Verts = TensorType["num_samples", "num_cams", "num_vertices", 3]  # 顶点
# Faces = TensorType["num_samples", "num_cams", "num_faces", 3]  # 面
# Proj = TensorType["num_samples", "num_cams", "height", "width"]  # 投影

# ------------------------------------------------------------------------------------- #
# MetricCallback 类
# ------------------------------------------------------------------------------------- #
class MetricCallback:
    def __init__(self, num_cams: int, num_classes: int, device: str):
        self.num_cams = num_cams

        # 初始化指标模块
        self.caption_metrics = {
            # "train": CaptionMetrics(num_classes),
            "test": CaptionMetrics(num_classes),
        }
        self.clatr_fd = {
            # "train": FrechetCLaTrDistance(),
            "test": FrechetCLaTrDistance(),
        }
        self.clatr_prdc = {
            # "train": ManifoldMetrics(distance="euclidean"),
            "test": ManifoldMetrics(distance="euclidean"),
        }
        self.clatr_score = {
            # "train": CLaTrScore(),
            "test": CLaTrScore(),
        }

        self.device = device
        self._move_to_device(device)

    def _move_to_device(self, device: str) -> None:
        for stage in ["test"]:
            self.clatr_fd[stage].to(device)
            self.clatr_prdc[stage].to(device)
            self.clatr_score[stage].to(device)

    # 更新字幕指标
    def update_caption_metrics(self, stage: str, pred: Traj, ref: Traj, mask: Mask):
        print("update_caption_metrics")
        self.caption_metrics[stage].update(pred, ref, mask)

    # 计算字幕指标
    def compute_caption_metrics(self, stage: str) -> Dict[str, float]:
        precision, recall, fscore = self.caption_metrics[stage].compute()
        self.caption_metrics[stage].reset()
        return {
            "captions/precision": precision,
            "captions/recall": recall,
            "captions/fscore": fscore,
        }

    # 更新CLaTr指标
    def update_clatr_metrics(self, stage: str, pred: Feat, ref: Feat, text: Feat):
        print("update_clatr_metrics")
        self.clatr_score[stage].update(pred, text)
        self.clatr_prdc[stage].update(ref, pred)
        self.clatr_fd[stage].update(ref, pred)

    # 计算CLaTr指标
    def compute_clatr_metrics(self, stage: str) -> Dict[str, float]:
        clatr_score = self.clatr_score[stage].compute()
        self.clatr_score[stage].reset()

        clatr_p, clatr_r, clatr_d, clatr_c = self.clatr_prdc[stage].compute()
        self.clatr_prdc[stage].reset()

        fcd = self.clatr_fd[stage].compute()
        self.clatr_fd[stage].reset()

        return {
            "clatr/clatr_score": clatr_score,
            "clatr/precision": clatr_p,
            "clatr/recall": clatr_r,
            "clatr/density": clatr_d,
            "clatr/coverage": clatr_c,
            "clatr/fcd": fcd,
        }

# ------------------------------------------------------------------------------------- #
# 计算指标
# ------------------------------------------------------------------------------------- #

def compute_metrics(
    num_cams: int,
    num_classes: int,
    device: str,
    pred_trajectories: TensorType["num_samples", "num_cams", 4, 4],
    ref_trajectories: TensorType["num_samples", "num_cams", 4, 4],
    pred_feats: TensorType["num_samples", "num_feats"],
    ref_feats: TensorType["num_samples", "num_feats"],
    text_feats: TensorType["num_samples", "num_feats"],
    mask: Mask  # 掩码
) -> Dict[str, float]:

    # 初始化 MetricCallback
    metric_callback = MetricCallback(num_cams=num_cams, num_classes=num_classes, device=device)
    # 更新并计算字幕指标
    for stage in ["test"]:
        metric_callback.update_caption_metrics(stage, pred_trajectories, ref_trajectories, mask)
    caption_metrics = metric_callback.compute_caption_metrics("test")  # 计算字幕指标

    # 更新并计算CLaTr指标
    for stage in ["test"]:
        metric_callback.update_clatr_metrics(stage, pred_feats, ref_feats, text_feats)
    clatr_metrics = metric_callback.compute_clatr_metrics("test")  # 计算CLaTr指标

    # 合并所有指标
    metrics = {**caption_metrics, **clatr_metrics}
    return metrics

# ------------------------------------------------------------------------------------- #
# 主函数：运行计算
# ------------------------------------------------------------------------------------- #


def get_feats(batch):
    pred_feats = batch["m_pred_latents"].to(device)
    ref_feat = batch["m_ref_latents"].to(device)
    text_feats = batch["t_latents"].to(device)
    pred_trajectories = batch["m_pred_matrices"].to(device)
    ref_trajectories = batch["m_ref_matrices"].to(device)
    masks = batch["masks"].to(device)
    
    return pred_feats, ref_feat, text_feats, pred_trajectories, ref_trajectories, masks

if __name__ == "__main__":
    # 这里是你的输入数据、、    # Set up argument parser
    parser = argparse.ArgumentParser(description="Set pred_path for model predictions")
    
    # Add pred_path as an argument
    parser.add_argument(
        '--pred_path', 
        type=str, 
        required=True,  # Makes the argument required
        help="Path to the predictions file"
    )

    # Parse the command line arguments
    args = parser.parse_args()

    # Access the pred_path from the parsed arguments
    pred_path = args.pred_path
    print(f"Prediction path: {pred_path}")
    
    set_random_seed(42)

    device = "cuda:0"  # 计算设备（示例为CPU）
    
    batch = np.load(pred_path, allow_pickle=True).item()
    # print(feats.keys())
    
    # print(feats["ref_matrices"][0])
    # print(feats["t_matrices"][0])
    # print(feats["m_ref_matrices"][0])
    # print(feats["m_pred_matrices"][0])
    
    pred_feats, ref_feat, text_feats, pred_trajectories, ref_trajectories, masks = get_feats(batch)
    # print(pred_feats.shape)
    # print(ref_feat.shape)
    # print(text_feats.shape)
    # print(pred_trajectories.shape)
    # print(ref_trajectories.shape)
    # print(masks.shape)
    num_cams = masks.shape[1]
    num_classes = 27 * 7  # 类别数（示例数据）
    
    # print(masks.shape)
    # print(masks)
    # num_sample = 100  # 摄像头数量（示例数据）
    # num_cams = 120  # 摄像头数量（示例数据）
    # num_feats = 256
    # pred_trajectories = torch.randn(num_sample, num_cams, 4, 4).to(device)  # 生成的摄像机轨迹（示例数据）, test数据集
    # ref_trajectories = torch.randn(num_sample, num_cams, 4, 4).to(device)  # 真实标签（示例数据), test数据集
    # pred_feats = torch.randn(num_sample, num_feats).to(device)  # 预测特征（示例数据）, test数据集
    # ref_feat = torch.randn(num_sample, num_feats).to(device)  # 参考特征（示例数据）, test数据集
    # text_feats = torch.randn(num_sample, num_feats).to(device)  # 文本特征（示例数据）, test数据集
    
    # caption_feats = ["This is a test caption"] * num_sample  # 文本描述（示例数据）
    # mask = torch.ones(num_sample, num_cams).to(device)  # 掩码（示例数据）

    # 计算指标
    metrics = compute_metrics(num_cams, num_classes, device, pred_trajectories, ref_trajectories, pred_feats, ref_feat, text_feats, masks)

    # 打印结果
    metrics_floats = {key: round(value.item(), 4) for key, value in metrics.items()}

    metrics_df = pd.DataFrame([metrics_floats])
    metrics_df.to_csv(f"results_epoch100/{pred_path.split('/')[-1][:-4]}.csv", index=False)  # 保存结果到CSV文件
    print("Metrics:", metrics_floats)