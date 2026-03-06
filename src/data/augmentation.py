"""
数据增强模块
支持多种异常生成算法用于提升模型泛化能力
"""

import random
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image


class CutPaste:
    """
    CutPaste 增强算法

    原理: 从正常图像中裁剪区域并粘贴到其他位置，
    模拟异常区域来训练模型的异常检测能力

    引用: "CutPaste: Self-Supervised Learning for Anomaly Detection" (CVPR 2021)
    """

    def __init__(
        self,
        crop_size: Tuple[float, float] = (0.1, 0.3),
        crop_scale: Tuple[float, float] = (0.5, 1.5),
        angle: Tuple[float, float] = (-45, 45),
        brightness: float = 0.5,
        alpha: float = 0.8,
    ):
        """
        Args:
            crop_size: 裁剪区域占图像大小的比例范围 (min, max)
            crop_scale: 裁剪后缩放比例范围
            angle: 旋转角度范围 (度)
            brightness: 亮度调整系数
            alpha: 混合透明度
        """
        self.crop_size = crop_size
        self.crop_scale = crop_scale
        self.angle = angle
        self.brightness = brightness
        self.alpha = alpha

    def __call__(
        self,
        image: Union[np.ndarray, Image.Image, torch.Tensor]
    ) -> Tuple[Union[np.ndarray, Image.Image], np.ndarray]:
        """
        应用 CutPaste 增强

        Args:
            image: 输入图像 (H, W, C) or (C, H, W)

        Returns:
            augmented_image: 增强后的图像
            mask: 异常区域掩码 (H, W)
        """
        # 转换为 numpy (H, W, C)
        image_np = self._to_numpy(image)
        h, w, c = image_np.shape

        # 生成裁剪区域
        crop_h = int(h * random.uniform(*self.crop_size))
        crop_w = int(w * random.uniform(*self.crop_size))

        # 随机裁剪位置
        y1 = random.randint(0, max(1, h - crop_h))
        x1 = random.randint(0, max(1, w - crop_w))

        # 裁剪
        crop = image_np[y1:y1+crop_h, x1:x1+crop_w].copy()

        # 随机缩放
        scale = random.uniform(*self.crop_scale)
        if scale != 1.0:
            new_h = int(crop_h * scale)
            new_w = int(crop_w * scale)
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            crop_h, crop_w = new_h, new_w

        # 随机旋转
        if self.angle[0] != 0 or self.angle[1] != 0:
            angle = random.uniform(*self.angle)
            center = (crop_w // 2, crop_h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            crop = cv2.warpAffine(crop, M, (crop_w, crop_h))

        # 调整大小适配目标区域
        y2 = random.randint(0, max(1, h - crop_h))
        x2 = random.randint(0, max(1, w - crop_w))

        # 创建掩码
        mask = np.zeros((h, w), dtype=np.float32)

        # 混合
        result = image_np.copy()
        crop_resized = cv2.resize(crop, (min(crop_w, w - x2), min(crop_h, h - y2)))

        crop_h_final = crop_resized.shape[0]
        crop_w_final = crop_resized.shape[1]

        y2 = min(y2, h - crop_h_final)
        x2 = min(x2, w - crop_w_final)

        # 应用增强
        result[y2:y2+crop_h_final, x2:x2+crop_w_final] = (
            self.alpha * crop_resized +
            (1 - self.alpha) * result[y2:y2+crop_h_final, x2:x2+crop_w_final]
        )

        # 更新掩码
        mask[y2:y2+crop_h_final, x2:x2+crop_w_final] = 1.0

        # 转换为原始格式
        result = self._from_numpy(result, image)

        return result, mask

    def _to_numpy(self, image):
        """转换为 HWC numpy 格式"""
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] in [1, 3]:
                image = image.permute(1, 2, 0)
            image = image.cpu().numpy()
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        return image

    def _from_numpy(self, image, original):
        """从 numpy 转换回原始格式"""
        if isinstance(original, torch.Tensor):
            if original.dim() == 3 and original.shape[0] in [1, 3]:
                image = torch.from_numpy(image).permute(2, 0, 1)
            else:
                image = torch.from_numpy(image)
            image = image.float() / 255.0
        elif isinstance(original, Image.Image):
            image = Image.fromarray(image)
        return image


class CopyPaste:
    """
    Copy-Paste 增强算法

    原理: 从一张图像复制区域粘贴到另一张图像，
    适合批量增强
    """

    def __init__(
        self,
        blend_mode: str = 'alpha',
        jitter: int = 5,
        min_size: float = 0.05,
        max_size: float = 0.3,
    ):
        """
        Args:
            blend_mode: 混合模式 ('alpha', 'poisson', 'none')
            jitter: 位置抖动范围
            min_size: 最小异常区域比例
            max_size: 最大异常区域比例
        """
        self.blend_mode = blend_mode
        self.jitter = jitter
        self.min_size = min_size
        self.max_size = max_size

    def __call__(
        self,
        source: np.ndarray,
        target: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用 Copy-Paste

        Args:
            source: 源图像 (异常来源)
            target: 目标图像

        Returns:
            result: 混合后的图像
            mask: 异常区域掩码
        """
        h, w = target.shape[:2]

        # 随机选择大小
        size = random.uniform(self.min_size, self.max_size)
        crop_h = int(h * size)
        crop_w = int(w * size)

        # 从源图像随机裁剪
        y1 = random.randint(0, max(1, source.shape[0] - crop_h))
        x1 = random.randint(0, max(1, source.shape[1] - crop_w))
        crop = source[y1:y1+crop_h, x1:x1+crop_w]

        # 随机位置（添加抖动）
        y2 = random.randint(-self.jitter, h - crop_h + self.jitter)
        x2 = random.randint(-self.jitter, w - crop_w + self.jitter)

        # 边界处理
        y2 = max(0, min(y2, h - 1))
        x2 = max(0, min(x2, w - 1))

        result = target.copy()
        mask = np.zeros((h, w), dtype=np.float32)

        # 计算有效区域
        crop_h = min(crop_h, h - y2)
        crop_w = min(crop_w, w - x2)

        if crop_h > 0 and crop_w > 0:
            crop = cv2.resize(crop, (crop_w, crop_h))

            if self.blend_mode == 'alpha':
                crop_float = crop.astype(np.float32)
                target_region = result[y2:y2+crop_h, x2:x2+crop_w].astype(np.float32)
                blended = 0.7 * crop_float + 0.3 * target_region
                result[y2:y2+crop_h, x2:x2+crop_w] = blended.astype(np.uint8)
            elif self.blend_mode == 'poisson':
                result = self._poisson_blend(crop, result, (y2, x2))
            else:
                result[y2:y2+crop_h, x2:x2+crop_w] = crop

            mask[y2:y2+crop_h, x2:x2+crop_w] = 1.0

        return result, mask

    def _poisson_blend(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        offset: Tuple[int, int]
    ) -> np.ndarray:
        """泊松融合"""
        y, x = offset
        h, w = src.shape[:2]

        # 确保在边界内
        y = max(0, min(y, dst.shape[0] - h))
        x = max(0, min(x, dst.shape[1] - w))

        # 创建圆形掩码
        mask = np.zeros((h, w, 1), dtype=np.float32)
        cv2.circle(mask, (w//2, h//2), min(w, h)//2, (1.0,), -1)

        # 融合
        result = dst.copy()
        src_float = src.astype(np.float32)
        dst_region = result[y:y+h, x:x+w].astype(np.float32)
        blended = src_float * mask + dst_region * (1 - mask)
        result[y:y+h, x:x+w] = blended.astype(np.uint8)

        return result


class NoiseInjection:
    """
    噪声注入增强

    模拟传感器噪声、传输噪声等常见异常
    """

    def __init__(
        self,
        noise_types: list = None,
        intensity_range: Tuple[float, float] = (0.05, 0.3),
    ):
        """
        Args:
            noise_types: 噪声类型列表 ('gaussian', 'salt', 'speckle', 'uniform')
            intensity_range: 噪声强度范围
        """
        self.noise_types = noise_types or ['gaussian', 'salt', 'speckle']
        self.intensity_range = intensity_range

    def __call__(
        self,
        image: Union[np.ndarray, Image.Image]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用噪声注入

        Args:
            image: 输入图像

        Returns:
            result: 添加噪声后的图像
            mask: 噪声区域掩码 (全1，表示整图区域)
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] in [1, 3]:
                image = image.permute(1, 2, 0).cpu().numpy()
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

        result = image.copy()
        h, w = result.shape[:2]

        # 随机选择噪声类型
        noise_type = random.choice(self.noise_types)
        intensity = random.uniform(*self.intensity_range)

        if noise_type == 'gaussian':
            noise = np.random.normal(0, intensity * 255, image.shape)
            result = np.clip(result + noise, 0, 255).astype(np.uint8)

        elif noise_type == 'salt':
            n_pixels = int(image.size * intensity * 0.5)
            coords = [np.random.randint(0, s, n_pixels) for s in image.shape[:2]]
            result[tuple(coords)] = 255

        elif noise_type == 'speckle':
            noise = np.random.randn(*image.shape) * intensity * 255
            result = np.clip(result + noise, 0, 255).astype(np.uint8)

        elif noise_type == 'uniform':
            noise = np.random.uniform(-intensity * 255, intensity * 255, image.shape)
            result = np.clip(result + noise, 0, 255).astype(np.uint8)

        # 全图作为异常区域
        mask = np.ones((h, w), dtype=np.float32)

        return result, mask


class BlurSharpness:
    """
    模糊/清晰度异常增强

    模拟对焦失败、运动模糊等常见工业检测异常
    """

    def __init__(
        self,
        blur_kernel_sizes: list = None,
        sharpness_kernels: list = None,
    ):
        """
        Args:
            blur_kernel_sizes: 模糊核大小列表
            sharpness_kernels: 锐化核类型
        """
        self.blur_kernel_sizes = blur_kernel_sizes or [3, 5, 7, 9]
        self.sharpness_kernels = sharpness_kernels or ['sharpen', 'unsharp']

    def __call__(
        self,
        image: Union[np.ndarray, Image.Image],
        anomaly_type: str = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用模糊/锐化

        Args:
            image: 输入图像
            anomaly_type: 指定异常类型 ('blur' 或 'sharpness')，随机选择如果为None

        Returns:
            result: 处理后的图像
            mask: 异常区域掩码 (全1)
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] in [1, 3]:
                image = image.permute(1, 2, 0).cpu().numpy()
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

        result = image.copy()
        h, w = result.shape[:2]

        # 随机选择异常类型
        if anomaly_type is None:
            anomaly_type = random.choice(['blur', 'sharpness'])

        if anomaly_type == 'blur':
            ksize = random.choice(self.blur_kernel_sizes)
            if ksize % 2 == 0:
                ksize += 1
            result = cv2.GaussianBlur(result, (ksize, ksize), 0)

        elif anomaly_type == 'sharpen':
            kernel = np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ], dtype=np.float32)
            result = cv2.filter2D(result, -1, kernel)

        elif anomaly_type == 'unsharp':
            blur = cv2.GaussianBlur(result, (5, 5), 0)
            result = cv2.addWeighted(result, 1.5, blur, -0.5, 0)

        # 全图作为异常区域
        mask = np.ones((h, w), dtype=np.float32)

        return result, mask


class AnomalyAugmentor:
    """
    统一的异常增强器

    整合所有增强方法，支持随机选择或指定类型
    """

    def __init__(
        self,
        methods: list = None,
        probabilities: list = None,
        **kwargs,
    ):
        """
        Args:
            methods: 增强方法列表
            probabilities: 各方法的使用概率
            **kwargs: 各方法的参数
        """
        self.methods = methods or ['cutpaste', 'noise', 'blur']
        self.probabilities = probabilities

        # 初始化各增强器
        self.augmentors = {
            'cutpaste': CutPaste(**kwargs.get('cutpaste', {})),
            'copy_paste': CopyPaste(**kwargs.get('copy_paste', {})),
            'noise': NoiseInjection(**kwargs.get('noise', {})),
            'blur': BlurSharpness(**kwargs.get('blur', {})),
        }

    def __call__(
        self,
        image: Union[np.ndarray, Image.Image, torch.Tensor],
        method: str = None,
    ) -> Tuple[Union[np.ndarray, Image.Image], np.ndarray]:
        """
        应用增强

        Args:
            image: 输入图像
            method: 指定方法，如果为None则随机选择

        Returns:
            augmented_image: 增强后的图像
            mask: 异常区域掩码
        """
        if method is None:
            method = random.choice(self.methods)

        if method == 'cutpaste':
            return self.augmentors['cutpaste'](image)
        elif method == 'copy_paste':
            # 需要两张图像
            raise ValueError("Copy-Paste requires source and target images")
        elif method == 'noise':
            return self.augmentors['noise'](image)
        elif method == 'blur':
            return self.augmentors['blur'](image)
        else:
            raise ValueError(f"Unknown augmentation method: {method}")

    def augment_batch(
        self,
        images: torch.Tensor,
        method: str = 'cutpaste',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量增强

        Args:
            images: (B, C, H, W) 张量
            method: 增强方法

        Returns:
            augmented: 增强后的张量
            masks: 异常掩码 (B, 1, H, W)
        """
        batch_size = images.shape[0]
        augmented_list = []
        masks_list = []

        for i in range(batch_size):
            img = images[i]  # (C, H, W)

            if method == 'cutpaste':
                aug_img, mask = self.augmentors['cutpaste'](img)
            elif method == 'noise':
                aug_img, mask = self.augmentors['noise'](img)
            elif method == 'blur':
                aug_img, mask = self.augmentors['blur'](img)
            else:
                aug_img, mask = img, torch.zeros(img.shape[-2:])

            augmented_list.append(aug_img)
            masks_list.append(torch.from_numpy(mask))

        augmented = torch.stack(augmented_list)
        masks = torch.stack(masks_list).unsqueeze(1)

        return augmented, masks


def create_anomaly_augmentor(
    method: str = 'cutpaste',
    **kwargs,
) -> Union[CutPaste, CopyPaste, NoiseInjection, BlurSharpness]:
    """创建异常增强器的便捷函数"""
    if method == 'cutpaste':
        return CutPaste(**kwargs)
    elif method == 'copy_paste':
        return CopyPaste(**kwargs)
    elif method == 'noise':
        return NoiseInjection(**kwargs)
    elif method == 'blur':
        return BlurSharpness(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
