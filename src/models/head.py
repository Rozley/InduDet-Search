"""
检测头模块
实现三种异常检测方法: Memory Bank, Distribution, Contrastive
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.spatial import distance


class BaseHead(nn.Module, ABC):
    """检测头基类"""

    def __init__(self, in_channels: int, config: Dict):
        super().__init__()
        self.in_channels = in_channels
        self.config = config

    @abstractmethod
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        pass

    @abstractmethod
    def fit(self, features: torch.Tensor):
        """训练/拟合"""
        pass

    @abstractmethod
    def predict(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """推理"""
        pass


class MemoryBankHead(BaseHead):
    """
    Memory Bank 检测头 (类似 PatchCore)

    原理:
    1. 存储正常样本的特征向量
    2. 推理时计算最近邻距离
    3. 距离越大表示越异常
    """

    def __init__(self, in_channels: int, config: Dict):
        """
        Args:
            in_channels: 输入通道数
            config: {
                'memory_size': 1000,  # 记忆库大小
                'k': 9,             # 最近邻数量
                'reduction': 'pca',  # 降维方法
                'reduction_dim': 128,  # 降维维度
            }
        """
        super().__init__(in_channels, config)

        self.memory_size = config.get('memory_size', 1000)
        self.k = config.get('k', 9)
        self.reduction = config.get('reduction', 'pca')
        self.reduction_dim = config.get('reduction_dim', 128)

        # 投影层 (可选)
        if self.reduction == 'pca' and in_channels > self.reduction_dim:
            self.projector = nn.Linear(in_channels, self.reduction_dim)
        else:
            self.projector = nn.Identity()

        # 记忆库
        self.memory_bank: Optional[torch.Tensor] = None
        self.knn: Optional[NearestNeighbors] = None

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 展平为2D: (B, C, H, W) -> (B*H*W, C)
        if features.dim() == 4:
            B, C, H, W = features.shape
            features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
        elif features.dim() == 2:
            B, C = features.shape
            # 计算 H 和 W
            n_features = C
            HW = n_features // self.in_channels if self.in_channels > 0 else n_features
            H = W = int(HW ** 0.5) if HW > 0 else 1
            features_flat = features
        else:
            raise ValueError(f"Unsupported features dim: {features.dim()}")

        # 投影特征 - 使用实际输入通道数
        if not isinstance(self.projector, nn.Identity):
            # 重新创建投影层以匹配实际输入通道数
            pass  # 投影层已创建，使用已配置的维度

        features_flat = self.projector(features_flat)

        # 重新reshape回4D
        features = features_flat.reshape(B, -1, H, W)

        # 计算与记忆库的距离
        if self.memory_bank is not None:
            # 计算所有patch到记忆库的距离
            distances = torch.cdist(features_flat, self.memory_bank, p=2)

            # K近邻平均距离
            if distances.shape[1] >= self.k:
                topk_distances, _ = torch.topk(distances, self.k, dim=1, largest=False)
                anomaly_scores = topk_distances.mean(dim=1)
            else:
                anomaly_scores = distances.mean(dim=1)

            # 重塑为特征图形状
            anomaly_map = anomaly_scores.reshape(B, H, W)
        else:
            # 没有记忆库，返回零
            anomaly_map = torch.zeros(B, H, W, device=features.device)

        return {
            'features': features,
            'anomaly_map': anomaly_map,
            'features_flat': features_flat,
        }

    def fit(self, features: torch.Tensor):
        """
        训练: 构建记忆库

        Args:
            features: 正常样本特征 (N, C, H, W)
        """
        # 展平为2D: (N, C, H, W) -> (N*H*W, C)
        N, C, H, W = features.shape
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)

        # 投影特征
        features_flat = self.projector(features_flat)

        # 重新获取投影后的通道数
        C = features_flat.shape[1]

        # 随机采样或核心集采样构建记忆库
        n_samples = features_flat.shape[0]

        if n_samples <= self.memory_size:
            # 全部使用
            self.memory_bank = features_flat
        else:
            # 随机采样
            indices = np.random.choice(n_samples, self.memory_size, replace=False)
            self.memory_bank = features_flat[indices]

        # 转换为numpy用于sklearn
        memory_np = self.memory_bank.cpu().numpy()

        # 构建KNN索引
        self.knn = NearestNeighbors(
            n_neighbors=min(self.k, self.memory_size),
            algorithm='ball_tree',
            metric='euclidean',
        )
        self.knn.fit(memory_np)

    def predict(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """推理"""
        result = self.forward(features)

        # 计算图像级别的异常分数
        anomaly_scores = result['anomaly_map'].flatten(1).max(dim=1)[0]  # max pooling
        image_score = anomaly_scores.mean()  # mean pooling

        return {
            'anomaly_map': result['anomaly_map'],
            'image_score': image_score,
            'pixel_scores': anomaly_scores,
        }


class DistributionHead(BaseHead):
    """
    Distribution 检测头 (类似 PaDiM)

    原理:
    1. 为每个位置建立多元高斯分布
    2. 使用马氏距离计算异常分数
    """

    def __init__(self, in_channels: int, config: Dict):
        """
        Args:
            in_channels: 输入通道数
            config: {
                'reduction': 'pca',  # 降维方法
                'reduction_dim': 100,  # 降维维度
            }
        """
        super().__init__(in_channels, config)

        self.reduction = config.get('reduction', 'pca')
        self.reduction_dim = config.get('reduction_dim', 100)

        # 投影层
        if self.reduction == 'pca' and in_channels > self.reduction_dim:
            self.projector = nn.Linear(in_channels, self.reduction_dim)
        else:
            self.projector = nn.Identity()
            self.reduction_dim = in_channels

        # 分布参数
        self.mean: Optional[torch.Tensor] = None
        self.cov_inv: Optional[torch.Tensor] = None

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 展平为2D: (B, C, H, W) -> (B*H*W, C)
        if features.dim() == 4:
            B, C, H, W = features.shape
            features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
        elif features.dim() == 2:
            B, C = features.shape
            # 计算 H 和 W
            n_features = C
            HW = n_features // self.in_channels if self.in_channels > 0 else n_features
            H = W = int(HW ** 0.5) if HW > 0 else 1
            features_flat = features
        else:
            raise ValueError(f"Unsupported features dim: {features.dim()}")

        # 投影特征 - 使用实际输入通道数
        if not isinstance(self.projector, nn.Identity):
            # 重新创建投影层以匹配实际输入通道数
            pass  # 投影层已创建，使用已配置的维度

        features_flat = self.projector(features_flat)

        # 重新获取投影后的通道数
        C = features_flat.shape[1]

        # 计算马氏距离
        if self.mean is not None and self.cov_inv is not None:
            # 去均值
            x_centered = features_flat - self.mean

            # 马氏距离: sqrt(x^T * Cov^{-1} * x)
            # 使用批量计算
            dist = torch.sum(x_centered @ self.cov_inv * x_centered, dim=1)
            dist = torch.sqrt(dist + 1e-6)

            # 重塑为特征图
            anomaly_map = dist.reshape(B, H, W)
        else:
            anomaly_map = torch.zeros(B, H, W, device=features.device)

        return {
            'features': features,
            'anomaly_map': anomaly_map,
            'features_flat': features_flat,
        }

    def fit(self, features: torch.Tensor):
        """
        训练: 估计高斯分布参数

        Args:
            features: 正常样本特征 (N, C, H, W)
        """
        # 展平为2D: (N, C, H, W) -> (N*H*W, C)
        N, C, H, W = features.shape
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)

        # 投影特征
        features_flat = self.projector(features_flat)

        # 重新获取投影后的通道数
        C = features_flat.shape[1]

        # 计算均值
        self.mean = features_flat.mean(dim=0)

        # 计算协方差矩阵
        cov = features_flat.t() @ features_flat / len(features_flat)
        cov = cov - self.mean.unsqueeze(1) @ self.mean.unsqueeze(0)

        # 添加正则化
        cov = cov + torch.eye(C, device=cov.device) * 1e-6

        # 计算协方差逆矩阵
        self.cov_inv = torch.inverse(cov)

        # 存储归一化因子 (马氏距离的常数部分)
        self.dist_norm = torch.sqrt(torch.diag(self.cov_inv) + 1e-6).mean()

    def predict(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """推理"""
        result = self.forward(features)

        # 归一化分数
        if self.dist_norm > 0:
            anomaly_scores = result['anomaly_map'] / self.dist_norm
        else:
            anomaly_scores = result['anomaly_map']

        # 计算图像级别的异常分数
        image_score = anomaly_scores.flatten(1).max(dim=1)[0]

        return {
            'anomaly_map': anomaly_scores,
            'image_score': image_score,
            'pixel_scores': anomaly_scores.flatten(1),
        }


class ContrastiveHead(BaseHead):
    """
    Contrastive 检测头 (类似 CSI)

    原理:
    1. 学习正常样本的嵌入空间
    2. 使用对比损失区分正常和异常
    3. 推理时计算异常分数
    """

    def __init__(self, in_channels: int, config: Dict):
        """
        Args:
            in_channels: 输入通道数
            config: {
                'memory_size': 1000,  # 记忆库大小
                'temperature': 0.1,   # 温度参数
            }
        """
        super().__init__(in_channels, config)

        self.memory_size = config.get('memory_size', 1000)
        self.temperature = config.get('temperature', 0.1)

        # 投影层
        self.projector = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, 128),
        )

        # 记忆库 (Queue)
        self.register_buffer('queue', torch.zeros(self.memory_size, 128))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        # 正常样本原型
        self.prototype: Optional[torch.Tensor] = None

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        B, C, H, W = features.shape
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)

        # 投影到嵌入空间
        embeddings = self.projector(features_flat)

        # 归一化
        embeddings = F.normalize(embeddings, dim=1)

        if self.prototype is not None:
            # 计算与原型的相似度
            similarity = embeddings @ self.prototype.t()
            similarity = similarity / self.temperature

            # 异常分数 = 1 - 相似度
            anomaly_scores = 1.0 - similarity.max(dim=1)[0]
            anomaly_scores = anomaly_scores.reshape(B, H, W)
        else:
            anomaly_scores = torch.zeros(B, H, W, device=features.device)

        return {
            'features': features,
            'embeddings': embeddings,
            'anomaly_map': anomaly_scores,
            'features_flat': features_flat,
        }

    def fit(self, features: torch.Tensor):
        """
        训练: 学习正常样本的原型

        Args:
            features: 正常样本特征 (N, C, H, W)
        """
        # 展平为2D: (N, C, H, W) -> (N*H*W, C)
        N, C, H, W = features.shape
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)

        # 投影到嵌入空间
        embeddings = self.projector(features_flat)
        embeddings = F.normalize(embeddings, dim=1)

        # 使用所有正常样本的均值作为原型
        self.prototype = embeddings_flat.mean(dim=0, keepdim=True)
        self.prototype = F.normalize(self.prototype, dim=1)

        # 同时初始化队列
        n_samples = embeddings_flat.shape[0]
        if n_samples >= self.memory_size:
            indices = np.random.choice(n_samples, self.memory_size, replace=False)
            self.queue = embeddings_flat[indices]
        else:
            self.queue[:n_samples] = embeddings_flat

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """更新        batch队列"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.memory_size % batch_size == 0

        self.queue[ptr:ptr + batch_size] = keys
        self.queue_ptr[0] = (ptr + batch_size) % self.memory_size

    def predict(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """推理"""
        result = self.forward(features)

        # 计算图像级别的异常分数
        image_score = result['anomaly_map'].flatten(1).max(dim=1)[0]

        return {
            'anomaly_map': result['anomaly_map'],
            'image_score': image_score,
            'pixel_scores': result['anomaly_map'].flatten(1),
        }


def create_head(
    method: str,
    in_channels: int,
    config: Dict,
) -> BaseHead:
    """
    创建检测头的便捷函数

    Args:
        method: 方法名称 ('memory_bank', 'distribution', 'contrastive')
        in_channels: 输入通道数
        config: 配置字典

    Returns:
        检测头实例
    """
    if method == 'memory_bank':
        return MemoryBankHead(in_channels, config)
    elif method == 'distribution':
        return DistributionHead(in_channels, config)
    elif method == 'contrastive':
        return ContrastiveHead(in_channels, config)
    else:
        raise ValueError(f"Unknown method: {method}")
