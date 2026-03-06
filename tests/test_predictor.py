"""
性能预测器测试
"""

import tempfile
import os

import numpy as np
import pytest

from src.search.predictor import (
    ArchitectureFeatureExtractor,
    PerformancePredictor,
    EarlyStopPredictor,
)


class TestArchitectureFeatureExtractor:
    """测试架构特征提取器"""

    def test_extract_features(self):
        """测试特征提取"""
        extractor = ArchitectureFeatureExtractor()

        architecture = {
            'backbone': 'ResNet18',
            'method': 'memory_bank',
            'attention': 'CBAM',
            'levels': [2, 3],
            'memory_size': 1000,
            'k': 9,
            'sampling': 'kcenter',
            'reduction': 'pca',
        }

        features = extractor.extract(architecture)

        assert isinstance(features, np.ndarray)
        assert len(features) == 16  # 特征数量
        assert features.shape == (16,)

    def test_default_architecture(self):
        """测试默认架构"""
        extractor = ArchitectureFeatureExtractor()

        architecture = {}
        features = extractor.extract(architecture)

        assert len(features) == 16

    def test_feature_names(self):
        """测试特征名称"""
        extractor = ArchitectureFeatureExtractor()
        names = extractor.get_feature_names()

        assert len(names) == 16
        assert 'log_params' in names
        assert 'method_complexity' in names


class TestPerformancePredictor:
    """测试性能预测器"""

    def test_predict_without_training(self):
        """测试未训练时的预测"""
        predictor = PerformancePredictor()

        architecture = {
            'backbone': 'ResNet18',
            'method': 'memory_bank',
            'attention': 'SE',
        }

        result = predictor.predict(architecture)

        assert 'predicted_auroc' in result
        assert 'estimated_latency_ms' in result
        # 未训练时使用默认预测
        assert result['predicted_auroc'] == 0.5

    def test_add_training_sample(self):
        """测试添加训练样本"""
        predictor = PerformancePredictor()

        architecture = {
            'backbone': 'ResNet18',
            'method': 'memory_bank',
            'attention': 'CBAM',
            'levels': [2, 3],
        }

        metrics = {
            'auroc': 0.95,
            'latency_ms': 30.0,
            'params': 10e6,
        }

        predictor.add_training_sample(architecture, metrics)

        assert len(predictor.training_data) == 1

    def test_train_with_sufficient_samples(self):
        """测试训练（有足够样本）"""
        predictor = PerformancePredictor()

        # 添加多个训练样本
        architectures = [
            {'backbone': 'ResNet18', 'method': 'memory_bank', 'attention': 'none', 'levels': [2, 3]},
            {'backbone': 'ResNet50', 'method': 'memory_bank', 'attention': 'none', 'levels': [2, 3]},
            {'backbone': 'WideResNet50', 'method': 'memory_bank', 'attention': 'SE', 'levels': [2, 3]},
            {'backbone': 'EfficientNet-B0', 'method': 'distribution', 'attention': 'CBAM', 'levels': [2, 3]},
            {'backbone': 'MobileNetV3', 'method': 'memory_bank', 'attention': 'none', 'levels': [2, 3]},
            {'backbone': 'ResNet18', 'method': 'distribution', 'attention': 'ECA', 'levels': [2, 3]},
            {'backbone': 'ResNet50', 'method': 'student_teacher', 'attention': 'SE', 'levels': [2, 3]},
            {'backbone': 'WideResNet50', 'method': 'memory_bank', 'attention': 'CBAM', 'levels': [2, 3]},
            {'backbone': 'EfficientNet-B3', 'method': 'memory_bank', 'attention': 'none', 'levels': [2, 3]},
            {'backbone': 'ResNet34', 'method': 'distribution', 'attention': 'SE', 'levels': [2, 3]},
        ]

        for i, arch in enumerate(architectures):
            metrics = {
                'auroc': 0.85 + i * 0.01,
                'latency_ms': 20.0 + i * 5,
                'params': 10e6 + i * 2e6,
            }
            predictor.add_training_sample(arch, metrics)

        # 训练
        predictor.train()

        assert predictor.is_trained

        # 预测
        result = predictor.predict(architectures[0])
        assert 'predicted_auroc' in result

    def test_batch_prediction(self):
        """测试批量预测"""
        predictor = PerformancePredictor()

        architectures = [
            {'backbone': 'ResNet18', 'method': 'memory_bank'},
            {'backbone': 'ResNet50', 'method': 'distribution'},
        ]

        results = predictor.predict(architectures)

        assert len(results) == 2
        for r in results:
            assert 'predicted_auroc' in r


class TestEarlyStopPredictor:
    """测试早停预测器"""

    def test_should_stop_below_threshold(self):
        """测试低于阈值时停止"""
        predictor = PerformancePredictor()
        early_stop = EarlyStopPredictor(predictor)

        architecture = {'backbone': 'ResNet18', 'method': 'memory_bank'}

        should_stop, reason = early_stop.should_stop(
            architecture,
            current_fidelity='low',
            current_auroc=0.4,  # 低于阈值 0.5
        )

        assert should_stop is True
        assert 'Below' in reason

    def test_should_continue_above_threshold(self):
        """测试高于阈值时继续"""
        predictor = PerformancePredictor()
        early_stop = EarlyStopPredictor(predictor)

        architecture = {'backbone': 'ResNet18', 'method': 'memory_bank'}

        should_stop, reason = early_stop.should_stop(
            architecture,
            current_fidelity='low',
            current_auroc=0.7,  # 高于阈值
        )

        assert should_stop is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
