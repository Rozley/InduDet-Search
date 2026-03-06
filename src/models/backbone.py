"""
Backbone 编码器模块
支持多种预训练网络作为特征提取器
"""

from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BaseEncoder(nn.Module, ABC):
    """编码器基类"""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.pretrained = pretrained

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征"""
        pass

    @abstractmethod
    def get_out_channels(self) -> List[int]:
        """获取各层输出通道数"""
        pass

    @abstractmethod
    def get_feature_levels(self, levels: List[int]) -> Dict[int, torch.Tensor]:
        """获取指定层级的特征"""
        pass


class ResNetEncoder(BaseEncoder):
    """ResNet 编码器"""

    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        progress: bool = True,
    ):
        super().__init__(pretrained)

        # 加载预训练模型
        if backbone == 'resnet18':
            self.model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            self.out_channels = [64, 128, 256, 512]
        elif backbone == 'resnet34':
            self.model = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
            self.out_channels = [64, 128, 256, 512]
        elif backbone == 'resnet50':
            self.model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            self.out_channels = [64, 256, 512, 1024]
        else:
            raise ValueError(f"Unknown ResNet backbone: {backbone}")

        # 移除最后的FC层和全局池化
        self.features = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4,
        )
        self.global_pool = nn.Identity()  # 使用patch级别的特征，不需要全局池化

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征"""
        return self.features(x)

    def get_out_channels(self) -> List[int]:
        return self.out_channels

    def get_feature_levels(self, levels: List[int]) -> Dict[int, torch.Tensor]:
        """获取指定层级的特征"""
        features = {}
        x = self.features[0](x)  # conv1
        x = self.features[1](x)  # bn1
        x = self.features[2](x)  # relu
        x = self.features[3](x)  # maxpool

        if 1 in levels:
            features[1] = self.features[4](x)  # layer1

        if 2 in levels:
            features[2] = self.features[5](x)  # layer2

        if 3 in levels:
            features[3] = self.features[6](x)  # layer3

        if 4 in levels:
            features[4] = self.features[7](x)  # layer4

        return features


class EfficientNetEncoder(BaseEncoder):
    """EfficientNet 编码器"""

    def __init__(
        self,
        backbone: str = 'efficientnet_b0',
        pretrained: bool = True,
        progress: bool = True,
    ):
        super().__init__(pretrained)

        if backbone == 'efficientnet_b0':
            self.model = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
            self.out_channels = [16, 24, 40, 80, 112, 192, 320, 1280]
        elif backbone == 'efficientnet_b3':
            self.model = models.efficientnet_b3(weights='IMAGENET1K_V1' if pretrained else None)
            self.out_channels = [24, 32, 48, 136, 232, 384, 1536, 1536]
        else:
            raise ValueError(f"Unknown EfficientNet backbone: {backbone}")

        # 特征提取器
        self.features = self.model.features
        self.avgpool = self.model.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征"""
        return self.features(x)

    def get_out_channels(self) -> List[int]:
        return self.out_channels[:-1]  # 去掉最后的1280

    def get_feature_levels(self, levels: List[int]) -> Dict[int, torch.Tensor]:
        """获取指定层级的特征"""
        features = {}
        x = self.features

        # EfficientNet没有明显的stage划分，使用索引
        for i, level in enumerate(levels):
            if level < len(self.out_channels):
                # 逐步提取特征
                for j in range(level + 1):
                    x = self.features[j](x) if j < len(self.features) else x
                features[level] = x

        return features


class MobileNetV3Encoder(BaseEncoder):
    """MobileNetV3 编码器"""

    def __init__(
        self,
        backbone: str = 'mobilenet_v3_large',
        pretrained: bool = True,
        progress: bool = True,
    ):
        super().__init__(pretrained)

        # 规范化 backbone 名称
        if backbone in ['mobilenetv3', 'mobilenet_v3', 'mobilenet_v3_large']:
            backbone = 'mobilenet_v3_large'
        elif backbone == 'mobilenet_v3_small':
            pass
        else:
            raise ValueError(f"Unknown MobileNet backbone: {backbone}")

        if backbone == 'mobilenet_v3_large':
            self.model = models.mobilenet_v3_large(weights='IMAGENET1K_V1' if pretrained else None)
        else:  # mobilenet_v3_small
            self.model = models.mobilenet_v3_small(weights='IMAGENET1K_V1' if pretrained else None)

        self.features = self.model.features
        self.out_channels = [16, 16, 24, 24, 40, 40, 80, 80, 112, 112, 960, 1280]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征"""
        return self.features(x)

    def get_out_channels(self) -> List[int]:
        return self.out_channels[:-2]  # 去掉最后的960和1280

    def get_feature_levels(self, levels: List[int]) -> Dict[int, torch.Tensor]:
        """获取指定层级的特征"""
        features = {}
        x = self.features

        for i, level in enumerate(levels):
            if level < len(self.out_channels):
                for j in range(level + 1):
                    x = self.features[j](x) if j < len(self.features) else x
                features[level] = x

        return features


class ViTEncoder(BaseEncoder):
    """Vision Transformer 编码器 (torchvision 0.25+)"""

    def __init__(
        self,
        backbone: str = 'vit_small',
        pretrained: bool = True,
        progress: bool = True,
    ):
        super().__init__(pretrained)

        # 模型名称映射
        MODEL_CONFIGS = {
            'vit_small': ('vit_b_16', 768),
            'vit_base': ('vit_l_16', 1024),
        }

        if backbone not in MODEL_CONFIGS:
            raise ValueError(f"Unknown ViT backbone: {backbone}")

        model_name, out_channels = MODEL_CONFIGS[backbone]
        model_fn = getattr(models, model_name, None)

        # 获取权重
        weights = getattr(models, f'{model_name.upper()}_WEIGHTS', None)
        default_weights = getattr(weights, 'IMAGENET1K_V1', None) if weights else None

        self.model = model_fn(weights=default_weights if pretrained else None)
        self.out_channels = out_channels

        # 获取 encoder 部分用于特征提取
        # torchvision 0.25+ ViT 结构
        self.encoder = self.model.encoder
        self.heads = self.model.heads
        self.patch_embed = self.model.patch_embed
        self.pos_embed = self.model.pos_embed
        self.seq_len = 14 * 14 + 1  # 14x14 patches + 1 class token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征"""
        B, C, H, W = x.shape

        # 提取 patch embeddings
        x = self.patch_embed(x)  # [B, num_patches, hidden_dim]
        cls_token = self.model.class_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed

        # 通过 encoder
        x = self.encoder(x)
        x = x.last_hidden_state  # [B, seq_len, hidden_dim]

        # 去掉 class token，重排为 2D 特征图
        side = 14  # 224 / 16 = 14
        features = x[:, 1:].transpose(1, 2).reshape(B, self.out_channels, side, side)

        return features
            # 3D: [B, seq_len, C] -> 去掉 class token，重排为 2D
            B, seq_len, C = x.shape
            side = 14  # 224 / 16 = 14
        return features

    def get_out_channels(self) -> List[int]:
        return [self.out_channels]

    def get_feature_levels(self, levels: List[int]) -> Dict[int, torch.Tensor]:
        """获取指定层级的特征"""
        x = self.forward(torch.zeros(1, 3, 224, 224))
        return {1: x}


def create_encoder(
    backbone: str,
    pretrained: bool = True,
    progress: bool = True,
) -> BaseEncoder:
    """
    创建编码器的便捷函数

    Args:
        backbone: backbone名称
        pretrained: 是否使用预训练权重
        progress: 是否显示下载进度

    Returns:
        编码器实例
    """
    backbone_lower = backbone.lower().replace('-', '_')

    # ViT 别名映射 (torchvision 0.25+)
    vit_aliases = {
        'vit_small': 'vit_small',
        'vit_base': 'vit_base',
    }
    if backbone_lower in vit_aliases:
        if not is_vit_available():
            raise ValueError(
                f"ViT backbone requires torchvision >= 0.12. "
                f"Your torchvision doesn't support ViT models."
            )
        backbone_lower = vit_aliases[backbone_lower]

    if 'resnet' in backbone_lower:
        return ResNetEncoder(backbone_lower, pretrained, progress)
    elif 'efficientnet' in backbone_lower:
        return EfficientNetEncoder(backbone_lower, pretrained, progress)
    elif 'mobilenet' in backbone_lower:
        return MobileNetV3Encoder(backbone_lower, pretrained, progress)
    elif 'vit' in backbone_lower:
        return ViTEncoder(backbone_lower, pretrained, progress)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")


def get_backbone_info(backbone: str) -> Dict[str, Any]:
    """获取backbone信息"""
    encoder = create_encoder(backbone, pretrained=False)
    return {
        'name': backbone,
        'out_channels': encoder.get_out_channels(),
        'params': sum(p.numel() for p in encoder.parameters()),
    }


def is_vit_available() -> bool:
    """检查 ViT 模型是否可用"""
    return hasattr(models, 'vit_b_16')
