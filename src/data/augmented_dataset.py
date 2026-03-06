"""
数据增强数据集
将异常增强集成到 PyTorch DataLoader
"""

import random
from typing import Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset


class AugmentedMVTecDataset(Dataset):
    """
    增强型 MVTec 数据集

    在训练时对正常样本应用异常增强，
    生成伪异常样本用于训练
    """

    def __init__(
        self,
        base_dataset: Dataset,
        augmentor: Optional[Callable] = None,
        anomaly_ratio: float = 0.3,
        augment_prob: float = 0.5,
        is_training: bool = True,
    ):
        """
        Args:
            base_dataset: 基础数据集
            augmentor: 异常增强器
            anomaly_ratio: 伪异常样本比例
            augment_prob: 正常样本被增强的概率
            is_training: 是否为训练模式
        """
        self.base_dataset = base_dataset
        self.augmentor = augmentor
        self.anomaly_ratio = anomaly_ratio
        self.augment_prob = augment_prob
        self.is_training = is_training

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        """
        返回:
            image: 增强后的图像
            label: 0=正常, 1=异常(伪异常)
            mask: 异常区域掩码
        """
        image, label, mask = self.base_dataset[idx]

        # 训练时：对正常样本进行增强生成伪异常
        if self.is_training and label == 0 and self.augmentor is not None:
            if random.random() < self.augment_prob:
                # 应用增强
                aug_image, aug_mask = self.augmentor(image)

                # 确保格式一致
                if isinstance(aug_image, torch.Tensor):
                    image = aug_image
                else:
                    # 转换回张量
                    if isinstance(aug_image, np.ndarray):
                        if aug_image.dtype != np.float32:
                            aug_image = aug_image.astype(np.float32) / 255.0
                        image = torch.from_numpy(aug_image).permute(2, 0, 1)

                if isinstance(aug_mask, np.ndarray):
                    mask = torch.from_numpy(aug_mask).unsqueeze(0)
                elif isinstance(aug_mask, torch.Tensor):
                    mask = aug_mask.unsqueeze(0) if aug_mask.dim() == 2 else aug_mask

                # 标记为伪异常
                label = 1

        return image, label, mask


class AugmentedDataLoader:
    """
    增强型数据加载器

    支持在线数据增强的数据加载
    """

    def __init__(
        self,
        base_dataset: Dataset,
        augmentor: Optional[Callable] = None,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        anomaly_ratio: float = 0.3,
        augment_prob: float = 0.5,
    ):
        """
        Args:
            base_dataset: 基础数据集
            augmentor: 异常增强器
            batch_size: 批次大小
            shuffle: 是否打乱
            num_workers: 工作进程数
            pin_memory: 是否固定内存
            anomaly_ratio: 伪异常比例
            augment_prob: 增强概率
        """
        self.augmented_dataset = AugmentedMVTecDataset(
            base_dataset=base_dataset,
            augmentor=augmentor,
            anomaly_ratio=anomaly_ratio,
            augment_prob=augment_prob,
            is_training=True,
        )

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.dataloader = torch.utils.data.DataLoader(
            self.augmented_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)


def create_augmented_datamodule(
    datamodule,
    augmentor=None,
    augment_prob: float = 0.5,
):
    """
    创建带增强的数据模块

    Args:
        datamodule: 基础数据模块 (MVTecDataModule)
        augmentor: 异常增强器
        augment_prob: 增强概率

    Returns:
        增强后的数据加载器
    """
    train_dataset = datamodule.get_train_dataset()

    return AugmentedDataLoader(
        base_dataset=train_dataset,
        augmentor=augmentor,
        batch_size=datamodule.batch_size,
        shuffle=True,
        num_workers=datamodule.num_workers,
        pin_memory=True,
        augment_prob=augment_prob,
    )
