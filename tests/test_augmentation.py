"""
数据增强模块测试
"""

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.augmentation import (
    CutPaste,
    CopyPaste,
    NoiseInjection,
    BlurSharpness,
    AnomalyAugmentor,
)


class TestCutPaste:
    """测试 CutPaste 增强"""

    def test_basic_functionality(self):
        """测试基本功能"""
        # 创建测试图像
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        # 创建 CutPaste 增强器
        augmentor = CutPaste(crop_size=(0.1, 0.3))

        # 应用增强
        result, mask = augmentor(image)

        # 验证输出
        assert result.shape == image.shape
        assert mask.shape == (256, 256)
        assert mask.dtype == np.float32
        assert np.sum(mask) > 0  # 应该有异常区域

    def test_with_pil_image(self):
        """测试 PIL 图像输入"""
        image = Image.new('RGB', (256, 256), color=(128, 128, 128))

        augmentor = CutPaste()
        result, mask = augmentor(image)

        assert isinstance(result, Image.Image)
        assert mask.shape == (256, 256)

    def test_with_tensor(self):
        """测试 PyTorch 张量输入"""
        image = torch.rand(3, 256, 256)

        augmentor = CutPaste()
        result, mask = augmentor(image)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 256, 256)
        assert mask.shape == (256, 256)


class TestNoiseInjection:
    """测试噪声注入"""

    def test_gaussian_noise(self):
        """测试高斯噪声"""
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        augmentor = NoiseInjection(noise_types=['gaussian'], intensity_range=(0.05, 0.1))
        result, mask = augmentor(image)

        assert result.shape == image.shape
        assert mask.shape == (256, 256)
        # 验证噪声已添加
        assert not np.array_equal(result, image)

    def test_salt_noise(self):
        """测试盐噪声"""
        image = np.full((256, 256, 3), 128, dtype=np.uint8)

        augmentor = NoiseInjection(noise_types=['salt'], intensity_range=(0.05, 0.1))
        result, mask = augmentor(image)

        # 应该有白色噪点
        assert np.any(result > 128)


class TestBlurSharpness:
    """测试模糊/锐化增强"""

    def test_blur(self):
        """测试模糊"""
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        augmentor = BlurSharpness()
        result, mask = augmentor(image, anomaly_type='blur')

        assert result.shape == image.shape
        assert mask.shape == (256, 256)

    def test_sharpness(self):
        """测试锐化"""
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        augmentor = BlurSharpness()
        result, mask = augmentor(image, anomaly_type='sharpen')

        assert result.shape == image.shape
        assert mask.shape == (256, 256)


class TestAnomalyAugmentor:
    """测试统一增强器"""

    def test_random_method(self):
        """测试随机方法选择"""
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        augmentor = AnomalyAugmentor(methods=['cutpaste', 'noise', 'blur'])
        result, mask = augmentor(image)

        assert result.shape == image.shape
        assert mask.shape == (256, 256)
        assert np.sum(mask) > 0

    def test_specified_method(self):
        """测试指定方法"""
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        augmentor = AnomalyAugmentor()
        result, mask = augmentor(image, method='noise')

        assert result.shape == image.shape

    def test_batch_augmentation(self):
        """测试批量增强"""
        images = torch.rand(4, 3, 256, 256)

        augmentor = AnomalyAugmentor()
        augmented, masks = augmentor.augment_batch(images, method='cutpaste')

        assert augmented.shape == images.shape
        assert masks.shape == (4, 1, 256, 256)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
