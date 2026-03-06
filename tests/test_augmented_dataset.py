"""
增强数据集测试
"""

import tempfile
import os

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


class DummyDataset(Dataset):
    """测试用虚拟数据集"""

    def __init__(self, size=10):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 返回 (image, label, mask)
        image = torch.rand(3, 224, 224)
        label = 0  # 正常样本
        mask = torch.zeros(1, 224, 224)
        return image, label, mask


from src.data.augmentation import CutPaste, NoiseInjection
from src.data.augmented_dataset import (
    AugmentedMVTecDataset,
    AugmentedDataLoader,
    create_augmented_datamodule,
)


class TestAugmentedMVTecDataset:
    """测试增强数据集"""

    def test_basic_functionality(self):
        """测试基本功能"""
        base_dataset = DummyDataset(size=10)
        augmentor = CutPaste()

        dataset = AugmentedMVTecDataset(
            base_dataset=base_dataset,
            augmentor=augmentor,
            augment_prob=1.0,  # 总是增强
            is_training=True,
        )

        assert len(dataset) == 10

        # 获取一个样本
        image, label, mask = dataset[0]

        assert image.shape == (3, 224, 224)
        assert mask.shape == (1, 224, 224)

    def test_with_noise_augmentor(self):
        """测试噪声增强器"""
        base_dataset = DummyDataset(size=10)
        augmentor = NoiseInjection()

        dataset = AugmentedMVTecDataset(
            base_dataset=base_dataset,
            augmentor=augmentor,
            augment_prob=0.5,
        )

        image, label, mask = dataset[0]

        assert image.shape == (3, 224, 224)

    def test_no_augmentation(self):
        """测试无增强"""
        base_dataset = DummyDataset(size=10)

        dataset = AugmentedMVTecDataset(
            base_dataset=base_dataset,
            augmentor=None,
            is_training=True,
        )

        image, label, mask = dataset[0]

        assert label == 0  # 仍然是正常样本


class TestAugmentedDataLoader:
    """测试增强数据加载器"""

    def test_dataloader_iteration(self):
        """测试数据加载器迭代"""
        base_dataset = DummyDataset(size=20)
        augmentor = CutPaste()

        loader = AugmentedDataLoader(
            base_dataset=base_dataset,
            augmentor=augmentor,
            batch_size=4,
            shuffle=True,
        )

        # 迭代一个批次
        batch = next(iter(loader))
        images, labels, masks = batch

        assert images.shape[0] == 4  # batch_size
        assert images.shape[1:] == (3, 224, 224)
        assert labels.shape[0] == 4

    def test_batch_size(self):
        """测试批次大小"""
        base_dataset = DummyDataset(size=20)
        augmentor = CutPaste()

        loader = AugmentedDataLoader(
            base_dataset=base_dataset,
            augmentor=augmentor,
            batch_size=8,
        )

        for images, labels, masks in loader:
            assert images.shape[0] <= 8
            break


class TestCreateAugmentedDatamodule:
    """测试创建增强数据模块"""

    def test_function(self):
        """测试函数"""
        base_dataset = DummyDataset(size=20)
        augmentor = CutPaste()

        # 模拟 datamodule
        class MockDataModule:
            def __init__(self):
                self.batch_size = 4
                self.num_workers = 0

            def get_train_dataset(self):
                return base_dataset

        mock_dm = MockDataModule()

        loader = create_augmented_datamodule(
            mock_dm,
            augmentor=augmentor,
            augment_prob=0.5,
        )

        assert loader is not None
        batch = next(iter(loader))
        assert batch[0].shape[0] <= mock_dm.batch_size


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
