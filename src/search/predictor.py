"""
性能预测器
基于架构特征快速预测最终性能
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler


class ArchitectureFeatureExtractor:
    """
    架构特征提取器

    将架构配置转换为数值特征向量
    """

    # 编码器特征映射
    BACKBONE_FEATURES = {
        'ResNet18': {'params': 11.7e6, 'flops': 1.8e9, 'depth': 18},
        'ResNet34': {'params': 21.8e6, 'flops': 3.7e9, 'depth': 34},
        'ResNet50': {'params': 25.6e6, 'flops': 4.1e9, 'depth': 50},
        'WideResNet50': {'params': 68.9e6, 'flops': 11.4e9, 'depth': 50},
        'EfficientNet-B0': {'params': 5.3e6, 'flops': 0.39e9, 'depth': 18},
        'EfficientNet-B3': {'params': 12.0e6, 'flops': 1.8e9, 'depth': 36},
        'MobileNetV3': {'params': 5.4e6, 'flops': 0.22e9, 'depth': 15},
        'ViT-Small': {'params': 22.0e6, 'flops': 4.5e9, 'depth': 12},
        'ViT-Base': {'params': 86.0e6, 'flops': 17.5e9, 'depth': 12},
    }

    # 检测头特征
    METHOD_FEATURES = {
        'memory_bank': {'complexity': 1.0, 'memory_intensive': True},
        'distribution': {'complexity': 0.8, 'memory_intensive': False},
        'student_teacher': {'complexity': 1.5, 'memory_intensive': True},
        'contrastive': {'complexity': 1.2, 'memory_intensive': False},
    }

    # 注意力模块特征
    ATTENTION_FEATURES = {
        'none': {'params_overhead': 0, 'flops_overhead': 0},
        'SE': {'params_overhead': 0.01, 'flops_overhead': 0.02},
        'CBAM': {'params_overhead': 0.03, 'flops_overhead': 0.05},
        'ECA': {'params_overhead': 0.005, 'flops_overhead': 0.01},
    }

    def extract(self, architecture: Dict) -> np.ndarray:
        """
        提取架构特征

        Args:
            architecture: 架构配置字典

        Returns:
            特征向量
        """
        features = []

        # 1. 编码器特征
        backbone = architecture.get('backbone', 'ResNet18')
        backbone_info = self.BACKBONE_FEATURES.get(backbone, {
            'params': 10e6, 'flops': 2e9, 'depth': 20
        })
        features.extend([
            np.log1p(backbone_info['params']),
            np.log1p(backbone_info['flops']),
            backbone_info['depth'],
        ])

        # 2. 检测头特征
        method = architecture.get('method', 'memory_bank')
        method_info = self.METHOD_FEATURES.get(method, {
            'complexity': 1.0, 'memory_intensive': False
        })
        features.extend([
            method_info['complexity'],
            1.0 if method_info['memory_intensive'] else 0.0,
        ])

        # 3. 注意力模块特征
        attention = architecture.get('attention', 'none')
        attn_info = self.ATTENTION_FEATURES.get(attention, {
            'params_overhead': 0, 'flops_overhead': 0
        })
        features.extend([
            attn_info['params_overhead'],
            attn_info['flops_overhead'],
        ])

        # 4. 特征层级
        levels = architecture.get('levels', [2, 3])
        features.extend([
            1.0 if 1 in levels else 0.0,
            1.0 if 2 in levels else 0.0,
            1.0 if 3 in levels else 0.0,
            1.0 if 4 in levels else 0.0,
            len(levels),
        ])

        # 5. 记忆库配置
        memory_size = architecture.get('memory_size', 1000)
        features.extend([
            np.log1p(memory_size),
            1.0 if architecture.get('sampling') == 'kcenter' else 0.0,
        ])

        # 6. k-NN 参数
        k = architecture.get('k', 9)
        features.append(k)

        # 7. 降维配置
        features.append(
            1.0 if architecture.get('reduction') == 'pca' else 0.0
        )

        return np.array(features, dtype=np.float32)

    def get_feature_names(self) -> List[str]:
        """获取特征名称"""
        return [
            'log_params',
            'log_flops',
            'backbone_depth',
            'method_complexity',
            'memory_intensive',
            'attention_params_overhead',
            'attention_flops_overhead',
            'use_layer_1',
            'use_layer_2',
            'use_layer_3',
            'use_layer_4',
            'num_levels',
            'log_memory_size',
            'use_kcenter_sampling',
            'k_nn',
            'use_pca_reduction',
        ]


class PerformancePredictor:
    """
    性能预测器

    基于架构特征预测 AUROC、延迟等指标
    """

    def __init__(
        self,
        model_type: str = 'gradient_boosting',
        feature_extractor: Optional[ArchitectureFeatureExtractor] = None,
    ):
        """
        Args:
            model_type: 模型类型 ('gradient_boosting', 'random_forest')
            feature_extractor: 特征提取器
        """
        self.model_type = model_type
        self.feature_extractor = feature_extractor or ArchitectureFeatureExtractor()
        self.scaler = StandardScaler()

        # 初始化模型
        if model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
            )

        self.is_trained = False
        self.training_data = []
        self._load_history()

    def _load_history(self):
        """加载历史训练数据"""
        history_path = Path('./knowledge/training_history.json')
        if history_path.exists():
            with open(history_path, 'r') as f:
                data = json.load(f)
                self.training_data = data.get('samples', [])

    def _save_history(self):
        """保存训练数据"""
        history_path = Path('./knowledge/training_history.json')
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, 'w') as f:
            json.dump({'samples': self.training_data}, f, indent=2)

    def add_training_sample(
        self,
        architecture: Dict,
        metrics: Dict,
    ):
        """
        添加训练样本

        Args:
            architecture: 架构配置
            metrics: 性能指标
        """
        sample = {
            'architecture': architecture,
            'auroc': metrics.get('auroc', 0.0),
            'latency_ms': metrics.get('latency_ms', 0.0),
            'params': metrics.get('params', 0),
        }
        self.training_data.append(sample)

    def train(self, force: bool = False):
        """
        训练预测模型

        Args:
            force: 是否强制重新训练
        """
        if self.is_trained and not force:
            return

        if len(self.training_data) < 10:
            print(f"Warning: Only {len(self.training_data)} samples, need at least 10 for training")
            return

        # 准备数据
        X = []
        y_auroc = []
        y_latency = []

        for sample in self.training_data:
            arch = sample['architecture']
            features = self.feature_extractor.extract(arch)
            X.append(features)
            y_auroc.append(sample['auroc'])
            y_latency.append(sample['latency_ms'])

        X = np.array(X)
        y_auroc = np.array(y_auroc)
        y_latency = np.array(y_latency)

        # 标准化
        X_scaled = self.scaler.fit_transform(X)

        # 训练 AUROC 预测模型
        self.model.fit(X_scaled, y_auroc)
        self.is_trained = True

        # 保存训练数据
        self._save_history()

        # 计算训练集分数
        train_score = self.model.score(X_scaled, y_auroc)
        print(f"Predictor trained. R² score: {train_score:.4f}")

    def predict(
        self,
        architecture: Union[Dict, List[Dict]],
    ) -> Union[Dict, List[Dict]]:
        """
        预测性能

        Args:
            architecture: 架构配置 (单个或列表)

        Returns:
            预测结果
        """
        if isinstance(architecture, dict):
            return self._predict_single(architecture)
        else:
            return [self._predict_single(arch) for arch in architecture]

    def _predict_single(self, architecture: Dict) -> Dict:
        """预测单个架构"""
        features = self.feature_extractor.extract(architecture)
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        if self.is_trained:
            auroc_pred = self.model.predict(features_scaled)[0]
            auroc_pred = np.clip(auroc_pred, 0.0, 1.0)
        else:
            # 使用默认预测
            auroc_pred = 0.5

        # 估计延迟 (简化版本)
        backbone = architecture.get('backbone', 'ResNet18')
        backbone_info = self.feature_extractor.BACKBONE_FEATURES.get(backbone, {
            'flops': 2e9
        })
        base_latency = backbone_info['flops'] / 1e9 * 10  # 粗略估计

        method = architecture.get('method', 'memory_bank')
        if method == 'memory_bank':
            base_latency *= 1.2

        attention = architecture.get('attention', 'none')
        if attention != 'none':
            base_latency *= 1.1

        return {
            'predicted_auroc': float(auroc_pred),
            'estimated_latency_ms': float(base_latency),
            'architecture': architecture,
        }

    def get_feature_importance(self) -> Dict:
        """获取特征重要性"""
        if not self.is_trained:
            return {}

        importances = self.model.feature_importances_
        feature_names = self.feature_extractor.get_feature_names()

        importance_dict = {
            name: float(imp)
            for name, imp in zip(feature_names, importances)
        }

        # 排序
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )

        return importance_dict


class EarlyStopPredictor:
    """
    早停预测器

    基于低保真度评估结果预测是否应该提前停止
    """

    def __init__(self, predictor: PerformancePredictor):
        self.predictor = predictor
        self.thresholds = {
            'low': 0.5,    # AUROC < 0.5 停止
            'medium': 0.7, # AUROC < 0.7 停止
            'high': 0.85,  # AUROC < 0.85 可考虑停止
        }

    def should_stop(
        self,
        architecture: Dict,
        current_fidelity: str,
        current_auroc: float,
    ) -> Tuple[bool, str]:
        """
        判断是否应该停止评估

        Args:
            architecture: 架构配置
            current_fidelity: 当前保真度 ('low', 'medium', 'high')
            current_auroc: 当前 AUROC

        Returns:
            (should_stop, reason)
        """
        # 基于当前保真度阈值判断
        threshold = self.thresholds.get(current_fidelity, 0.0)

        if current_auroc < threshold:
            return True, f"Below {current_fidelity} threshold {threshold}"

        # 使用预测器预测最终性能
        prediction = self.predictor.predict(architecture)

        predicted_final = prediction['predicted_auroc']

        # 如果预测最终性能低于阈值，提前停止
        if predicted_final < threshold and current_fidelity != 'high':
            return True, f"Predicted final AUROC {predicted_final:.4f} below threshold"

        return False, "Continue evaluation"


def create_predictor(
    model_type: str = 'gradient_boosting',
) -> Tuple[PerformancePredictor, ArchitectureFeatureExtractor]:
    """创建性能预测器的便捷函数"""
    extractor = ArchitectureFeatureExtractor()
    predictor = PerformancePredictor(model_type, extractor)
    return predictor, extractor
