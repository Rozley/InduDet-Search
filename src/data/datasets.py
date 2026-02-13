"""
MVTec AD 数据集加载器
支持图像分类和像素级异常检测
"""

import os
import glob
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MVTecDataset(Dataset):
    """
    MVTec AD 数据集

    数据集结构:
    mvtec/
    ├── bottle/
    │   ├── train/
    │   │   └── good/
    │   │       ├── 000.png
    │   │       └── ...
    │   ├── test/
    │   │   ├── good/
    │   │   │   └── ...
    │   │   ├── broken_large/
    │   │   │   └── ...
    │   │   └── ...
    │   └── ground_truth/
    │       ├── broken_large/
    │       │   └── ...
    │       └── ...
    """

    def __init__(
        self,
        root: str,
        category: str = 'bottle',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        train: bool = True,
        anomaly_ratio: float = 0.0,
    ):
        """
        Args:
            root: 数据集根目录
            category: 物体类别
            transform: 图像变换
            target_transform: 标签变换
            train: 是否为训练集 (True返回正常样本, False返回测试样本)
            anomaly_ratio: 异常样本比例 (用于半监督训练)
        """
        self.root = Path(root) / category
        self.category = category
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.anomaly_ratio = anomaly_ratio

        self.image_paths = []
        self.labels = []
        self.mask_paths = []

        self._load_data()

    def _load_data(self):
        """加载数据路径"""
        if self.train:
            # 训练集: 只有正常样本
            train_good_dir = self.root / 'train' / 'good'
            if train_good_dir.exists():
                for img_path in sorted(train_good_dir.glob('*.png')):
                    self.image_paths.append(img_path)
                    self.labels.append(0)  # 0 = 正常
                    self.mask_paths.append(None)
        else:
            # 测试集: 包含正常和异常样本
            test_dir = self.root / 'test'
            if test_dir.exists():
                for subdir in sorted(test_dir.iterdir()):
                    if subdir.is_dir():
                        is_normal = subdir.name == 'good'
                        label = 0 if is_normal else 1  # 0=正常, 1=异常

                        # 加载图像
                        for img_path in sorted(subdir.glob('*.png')):
                            self.image_paths.append(img_path)
                            self.labels.append(label)
                            self.mask_paths.append(None)

                        # 加载ground truth masks
                        gt_dir = self.root / 'ground_truth' / subdir.name
                        if gt_dir.exists():
                            for mask_path in sorted(gt_dir.glob('*.png')):
                                # 找到对应的mask
                                base_name = mask_path.stem
                                idx = self._find_mask_index(base_name)
                                if idx is not None:
                                    self.mask_paths[idx] = mask_path

    def _find_mask_index(self, base_name: str) -> Optional[int]:
        """根据mask文件名查找对应的索引"""
        for i, path in enumerate(self.image_paths):
            if path.stem == base_name:
                return i
        return None

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
        """
        返回:
            image: 图像张量 (C, H, W)
            label: 标签 (0=正常, 1=异常)
            mask: 异常掩码 (仅测试时使用)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        mask_path = self.mask_paths[idx]

        # 加载图像
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # 加载mask
        mask = None
        if mask_path and mask_path.exists():
            mask = Image.open(mask_path).convert('L')
            if self.transform:
                # Mask使用最近邻插值
                mask_transform = transforms.Compose([
                    transforms.Resize((image.shape[1], image.shape[2]), interpolation=transforms.InterpolationMode.NEAREST),
                    transforms.ToTensor(),
                ])
                mask = mask_transform(mask)

        return image, label, mask


class MVTecDataModule:
    """MVTec 数据模块 - 封装数据集和DataLoader"""

    def __init__(
        self,
        data_dir: str = './data/mvtec',
        category: str = 'bottle',
        image_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        self.data_dir = data_dir
        self.category = category
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        # 默认图像变换
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def get_train_dataset(self, anomaly_ratio: float = 0.0) -> MVTecDataset:
        """获取训练数据集"""
        return MVTecDataset(
            root=self.data_dir,
            category=self.category,
            transform=self.train_transform,
            train=True,
            anomaly_ratio=anomaly_ratio,
        )

    def get_test_dataset(self) -> MVTecDataset:
        """获取测试数据集"""
        return MVTecDataset(
            root=self.data_dir,
            category=self.category,
            transform=self.test_transform,
            train=False,
        )

    def train_dataloader(self, anomaly_ratio: float = 0.0) -> DataLoader:
        """获取训练DataLoader"""
        dataset = self.get_train_dataset(anomaly_ratio)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        """获取测试DataLoader"""
        dataset = self.get_test_dataset()
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @staticmethod
    def get_available_categories(data_dir: str) -> List[str]:
        """获取可用的类别列表"""
        dir_path = Path(data_dir)
        if not dir_path.exists():
            return []
        return [d.name for d in dir_path.iterdir() if d.is_dir()]


class MVTecSubset(Dataset):
    """MVTec 数据子集 - 用于多保真度评估"""

    def __init__(
        self,
        dataset: MVTecDataset,
        ratio: float = 0.1,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.ratio = ratio
        self.indices = self._create_subset(ratio, shuffle)

    def _create_subset(self, ratio: float, shuffle: bool) -> List[int]:
        """创建子集索引"""
        n_samples = len(self.dataset)
        n_subset = max(1, int(n_samples * ratio))

        indices = list(range(n_samples))
        if shuffle:
            np.random.shuffle(indices)

        return indices[:n_subset]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
        real_idx = self.indices[idx]
        return self.dataset[real_idx]


def create_mvtec_datamodule(
    data_dir: str,
    category: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
) -> MVTecDataModule:
    """创建MVTec数据模块的便捷函数"""
    return MVTecDataModule(
        data_dir=data_dir,
        category=category,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )
