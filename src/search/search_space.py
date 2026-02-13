"""
搜索空间定义
简化版搜索空间 - 5个维度 x 405种组合
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# ==================== 简化的搜索空间 ====================
SEARCH_SPACE = {
    # 编码器选择（5选1）
    'backbone': [
        'ResNet18',
        'ResNet50',
        'EfficientNet-B0',
        'MobileNetV3',
        'ViT-Small',
    ],

    # 特征层级（3选1）
    'feature_levels': [
        'L2',           # 只用第2层
        'L2+L3',        # 用第2、3层
        'L2+L3+L4',     # 用第2、3、4层
    ],

    # 检测方法（3选1）
    'method': [
        'memory_bank',   # PatchCore风格
        'distribution',  # PaDiM风格
        'contrastive',  # CSI风格
    ],

    # 记忆库大小（3选1）
    'memory_size': [500, 1000, 2000],

    # k-NN参数（3选1）
    'k': [1, 5, 9],
}


def get_search_space() -> Dict[str, List[str]]:
    """获取搜索空间"""
    return SEARCH_SPACE.copy()


def get_search_space_size() -> int:
    """计算搜索空间大小"""
    size = 1
    for options in SEARCH_SPACE.values():
        size *= len(options)
    return size


def sample_random_config() -> Dict[str, Any]:
    """随机采样一个配置"""
    config = {}
    for key, options in SEARCH_SPACE.items():
        config[key] = np.random.choice(options)
    return config


def config_to_vector(config: Dict[str, Any]) -> List[float]:
    """将配置转换为向量（用于贝叶斯优化）"""
    vector = []
    keys = list(SEARCH_SPACE.keys())

    for key in keys:
        options = SEARCH_SPACE[key]
        value = config[key]
        # one-hot编码
        vec = [1.0 if v == value else 0.0 for v in options]
        vector.extend(vec)

    return vector


def vector_to_config(vector: List[float]) -> Dict[str, Any]:
    """将向量转换为配置"""
    config = {}
    keys = list(SEARCH_SPACE.keys())
    idx = 0

    for key in keys:
        options = SEARCH_SPACE[key]
        n_options = len(options)
        vec = vector[idx:idx + n_options]
        idx += n_options

        # 找到最大值对应的选项
        best_idx = np.argmax(vec)
        config[key] = options[best_idx]

    return config


def get_config_hash(config: Dict[str, Any]) -> str:
    """获取配置的哈希值"""
    items = sorted(config.items())
    return hash(tuple(items))


def compare_configs(config1: Dict[str, Any], config2: Dict[str, Any]) -> bool:
    """比较两个配置是否相同"""
    return get_config_hash(config1) == get_config_hash(config2)


class SearchSpace:
    """搜索空间类"""

    def __init__(self, custom_space: Optional[Dict[str, List[Any]]] = None):
        """
        Args:
            custom_space: 自定义搜索空间（覆盖默认空间）
        """
        if custom_space:
            self.space = {**SEARCH_SPACE, **custom_space}
        else:
            self.space = SEARCH_SPACE.copy()

    @property
    def backbones(self) -> List[str]:
        return self.space['backbone']

    @property
    def feature_levels(self) -> List[str]:
        return self.space['feature_levels']

    @property
    def methods(self) -> List[str]:
        return self.space['method']

    @property
    def memory_sizes(self) -> List[int]:
        return self.space['memory_size']

    @property
    def k_values(self) -> List[int]:
        return self.space['k']

    @property
    def keys(self) -> List[str]:
        return list(self.space.keys())

    def sample(self) -> Dict[str, Any]:
        """随机采样一个配置"""
        return sample_random_config()

    def size(self) -> int:
        """获取搜索空间大小"""
        return get_search_space_size()

    def encode(self, config: Dict[str, Any]) -> List[float]:
        """编码配置"""
        return config_to_vector(config)

    def decode(self, vector: List[float]) -> Dict[str, Any]:
        """解码配置"""
        return vector_to_config(vector)


# ==================== ResNet 特征层级映射 ====================
RESNET_LEVELS = {
    'L2': [2],           # layer2 输出
    'L2+L3': [2, 3],     # layer2 + layer3 输出
    'L2+L3+L4': [2, 3, 4],  # layer2 + layer3 + layer4 输出
}

# ResNet 各层输出通道数
RESNET_CHANNELS = {
    'ResNet18': [64, 128, 256, 512],
    'ResNet50': [64, 256, 512, 1024],
    'EfficientNet-B0': [16, 24, 40, 80, 112, 192, 320],
    'MobileNetV3': [16, 16, 24, 24, 40, 40, 80, 80, 112, 112, 960, 1280],
    'ViT-Small': [384],  # ViT使用单独的投影层
}


def get_feature_levels(levels_str: str) -> List[int]:
    """获取特征层级列表"""
    return RESNET_LEVELS.get(levels_str, [2, 3])


def get_backbone_channels(backbone: str) -> List[int]:
    """获取backbone各层通道数"""
    return RESNET_CHANNELS.get(backbone, [64, 128, 256, 512])
