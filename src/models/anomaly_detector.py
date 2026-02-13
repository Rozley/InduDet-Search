"""
完整异常检测模型
组合编码器和检测头
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import create_encoder
from .head import create_head


class AnomalyDetector(nn.Module):
    """
    完整异常检测模型

    架构:
    1. Backbone: 特征编码器
    2. Head: 异常检测头
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: {
                'backbone': 'ResNet50',
                'feature_levels': 'L2+L3',
                'method': 'memory_bank',
                'memory_size': 1000,
                'k': 9,
            }
        """
        super().__init__()

        self.config = config
        self.backbone_name = config.get('backbone', 'ResNet50')
        self.feature_levels_str = config.get('feature_levels', 'L2+L3')
        self.method = config.get('method', 'memory_bank')

        # 解析特征层级
        self.feature_levels = self._parse_levels(self.feature_levels_str)

        # 创建编码器
        self.encoder = create_encoder(
            backbone=self.backbone_name,
            pretrained=True,
        )

        # 获取特征通道数
        backbone_channels = self.encoder.get_out_channels()
        self.in_channels = self._get_in_channels(backbone_channels)

        # 创建检测头
        head_config = {
            'memory_size': config.get('memory_size', 1000),
            'k': config.get('k', 9),
        }
        self.head = create_head(
            method=self.method,
            in_channels=self.in_channels,
            config=head_config,
        )

        # 注册缓冲区用于存储原始特征（用于测试时使用）
        self.register_buffer('features_buffer', torch.zeros(1, 1))

    def _parse_levels(self, levels_str: str) -> List[int]:
        """解析特征层级字符串"""
        if levels_str == 'L2':
            return [2]
        elif levels_str == 'L2+L3':
            return [2, 3]
        elif levels_str == 'L2+L3+L4':
            return [2, 3, 4]
        else:
            return [2, 3]

    def _get_in_channels(self, backbone_channels: List[int]) -> int:
        """获取输入到检测头的通道数"""
        # 获取选定层级的最大通道数
        max_level = max(self.feature_levels)
        # levels是1-based，需要减1
        idx = max_level - 1
        if idx < len(backbone_channels):
            return backbone_channels[idx]
        return backbone_channels[-1]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入图像 (B, C, H, W)

        Returns:
            {
                'features': 提取的特征,
                'anomaly_map': 异常分数图,
                'image_score': 图像级异常分数,
            }
        """
        # 提取特征
        features = self.encoder(x)

        # 获取指定层级的特征
        level_features = self._get_level_features(features)

        # 检测头
        head_output = self.head(level_features)

        return {
            'features': head_output.get('features', level_features),
            'anomaly_map': head_output['anomaly_map'],
            'image_score': head_output.get('image_score', head_output['anomaly_map'].mean()),
        }

    def _get_level_features(self, features: torch.Tensor) -> torch.Tensor:
        """获取指定层级的特征"""
        # 直接返回 encoder 的原始输出，保持 4D (B, C, H, W)
        return features

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征（用于训练前）"""
        return self.encoder(x)

    def fit(self, train_loader):
        """
        训练模型（无监督，只用正常样本）

        Args:
            train_loader: 训练数据加载器
        """
        # 提取正常样本的特征
        all_features = []

        self.encoder.eval()
        with torch.no_grad():
            for batch in train_loader:
                images = batch[0]  # (B, C, H, W)
                images = images.cuda()

                features = self.extract_features(images)
                all_features.append(features)

        all_features = torch.cat(all_features, dim=0)

        # 训练检测头
        self.head.fit(all_features)

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        推理

        Args:
            x: 输入图像 (B, C, H, W)

        Returns:
            {
                'anomaly_map': 异常分数图 (B, H, W),
                'image_score': 图像级异常分数 (B,),
            }
        """
        output = self.forward(x)

        return {
            'anomaly_map': output['anomaly_map'],
            'image_score': output['image_score'],
        }

    def compute_score(self, anomaly_map: torch.Tensor) -> float:
        """计算单个图像的异常分数"""
        return anomaly_map.mean().item()

    def count_parameters(self) -> int:
        """统计参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'backbone': self.backbone_name,
            'feature_levels': self.feature_levels_str,
            'method': self.method,
            'in_channels': self.in_channels,
            'params': self.count_parameters(),
        }


def create_anomaly_detector(config: Dict) -> AnomalyDetector:
    """
    创建异常检测模型的便捷函数

    Args:
        config: 模型配置字典

    Returns:
        AnomalyDetector实例
    """
    return AnomalyDetector(config)
