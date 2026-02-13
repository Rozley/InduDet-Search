"""
配置加载器
支持YAML配置文件加载和环境变量解析
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """配置类 - 单例模式"""

    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._config:
            self.load_default()

    @classmethod
    def load(cls, config_path: str = None) -> 'Config':
        """加载配置文件"""
        instance = cls()
        if config_path:
            instance.load_from_file(config_path)
        return instance

    @classmethod
    def get_instance(cls) -> 'Config':
        """获取配置单例"""
        return cls.load()

    def load_default(self):
        """加载默认配置"""
        default_config = {
            'dataset': {
                'name': 'mvtec',
                'data_dir': './data/mvtec',
                'image_size': 224,
                'num_channels': 3,
                'train_batch_size': 32,
                'eval_batch_size': 64,
                'num_workers': 4,
            },
            'search_space': {
                'backbone': ['ResNet18', 'ResNet50', 'EfficientNet-B0', 'MobileNetV3', 'ViT-Small'],
                'feature_levels': ['L2', 'L2+L3', 'L2+L3+L4'],
                'method': ['memory_bank', 'distribution', 'contrastive'],
                'memory_size': [500, 1000, 2000],
                'k': [1, 5, 9],
            },
            'search_strategy': {
                'name': 'incremental',
                'n_random': 50,
                'n_total': 200,
                'n_candidates': 100,
            },
            'evaluator': {
                'fidelity_levels': [
                    {'level': 'low', 'epochs': 1, 'data_ratio': 0.1, 'threshold': 0.60},
                    {'level': 'medium', 'epochs': 10, 'data_ratio': 0.5, 'threshold': 0.75},
                    {'level': 'high', 'epochs': 50, 'data_ratio': 1.0, 'threshold': 0.0},
                ],
            },
            'experience': {
                'storage_type': 'json',
                'storage_path': './results/experiences.json',
                'max_cases': 1000,
                'success_threshold': 0.85,
                'failure_threshold': 0.50,
            },
            'llm': {
                'provider': 'minimax',
                'api_key': '',
                'base_url': 'https://api.minimax.chat/v1',
                'model': 'MiniMax-M2.1',
                'max_tokens': 2048,
                'temperature': 0.7,
                'use_for_suggestion': True,
            },
            'training': {
                'optimizer': 'adam',
                'learning_rate': 1e-4,
                'weight_decay': 1e-4,
                'scheduler': 'cosine',
                'warmup_epochs': 5,
                'max_epochs': 100,
            },
            'resources': {
                'max_time_hours': 24,
                'max_gpu_memory_gb': 24,
                'cuda_visible_devices': '0',
            },
            'output': {
                'save_dir': './results',
                'checkpoint_interval': 20,
                'log_interval': 10,
                'save_best': True,
                'save_all': False,
            },
            'device': {
                'type': 'cuda',
                'precision': 'fp32',
            },
        }
        self._config = default_config

    def load_from_file(self, config_path: str):
        """从YAML文件加载配置"""
        path = Path(config_path)
        if not path.exists():
            print(f"Warning: Config file {config_path} not found, using defaults")
            self.load_default()
            return

        with open(path, 'r', encoding='utf-8') as f:
            file_config = yaml.safe_load(f)

        if file_config:
            # 递归更新配置
            self._update_nested(self._config, file_config)

    def _update_nested(self, base: Dict, update: Dict):
        """递归更新嵌套字典"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._update_nested(base[key], value)
            else:
                base[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值 - 支持点号分隔的路径"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any):
        """设置配置值 - 支持点号分隔的路径"""
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    @property
    def dataset(self) -> Dict:
        """获取数据集配置"""
        return self._config.get('dataset', {})

    @property
    def search_space(self) -> Dict:
        """获取搜索空间配置"""
        return self._config.get('search_space', {})

    @property
    def search_strategy(self) -> Dict:
        """获取搜索策略配置"""
        return self._config.get('search_strategy', {})

    @property
    def evaluator(self) -> Dict:
        """获取评估器配置"""
        return self._config.get('evaluator', {})

    @property
    def experience(self) -> Dict:
        """获取经验系统配置"""
        return self._config.get('experience', {})

    @property
    def llm(self) -> Dict:
        """获取LLM配置"""
        return self._config.get('llm', {})

    @property
    def training(self) -> Dict:
        """获取训练配置"""
        return self._config.get('training', {})

    @property
    def resources(self) -> Dict:
        """获取资源限制配置"""
        return self._config.get('resources', {})

    @property
    def output(self) -> Dict:
        """获取输出配置"""
        return self._config.get('output', {})

    @property
    def device(self) -> Dict:
        """获取设备配置"""
        return self._config.get('device', {})

    def to_dict(self) -> Dict:
        """将配置转换为字典"""
        return self._config.copy()

    def save(self, path: str):
        """保存配置到文件"""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)


def load_config(config_path: str = None) -> Config:
    """加载配置的便捷函数"""
    return Config.load(config_path)
