"""
奖励函数测试
"""

import pytest
import numpy as np

from src.search.reward import (
    MultiDimensionReward,
    EfficiencyConstraintReward,
    SRLLMReward,
    compute_reward,
)


class TestMultiDimensionReward:
    """测试多维度奖励"""

    def test_basic_computation(self):
        """测试基本计算"""
        reward_fn = MultiDimensionReward()

        metrics = {
            'auroc': 0.95,
            'f1': 0.90,
            'latency_ms': 30.0,
            'params': 10e6,
        }

        reward, scores = reward_fn.compute(metrics)

        assert reward > 0
        assert 'auroc' in scores
        assert 'latency' in scores
        assert 'params' in scores

    def test_auroc_normalization(self):
        """测试 AUROC 标准化"""
        reward_fn = MultiDimensionReward()

        # AUROC = 0.5 -> score = 0
        _, scores = reward_fn.compute({'auroc': 0.5})
        assert scores['auroc'] == 0.0

        # AUROC = 0.75 -> score = 0.5
        _, scores = reward_fn.compute({'auroc': 0.75})
        assert scores['auroc'] == 0.5

        # AUROC = 1.0 -> score = 1.0
        _, scores = reward_fn.compute({'auroc': 1.0})
        assert scores['auroc'] == 1.0

    def test_latency_normalization(self):
        """测试延迟标准化"""
        reward_fn = MultiDimensionReward()

        # 延迟越小分数越高
        _, scores_low = reward_fn.compute({'auroc': 0.9, 'latency_ms': 10})
        _, scores_high = reward_fn.compute({'auroc': 0.9, 'latency_ms': 100})

        assert scores_low['latency'] > scores_high['latency']

    def test_weights(self):
        """测试自定义权重"""
        weights = {'auroc': 1.0, 'latency': 0.0, 'params': 0.0}
        reward_fn = MultiDimensionReward(weights=weights)

        metrics = {'auroc': 0.9, 'latency_ms': 100, 'params': 100e6}
        reward, scores = reward_fn.compute(metrics)

        # 奖励应该只来自 auroc
        assert scores['latency'] == 0.0
        assert scores['params'] == 0.0


class TestEfficiencyConstraintReward:
    """测试效率约束奖励"""

    def test_meets_constraints(self):
        """测试满足约束"""
        reward_fn = EfficiencyConstraintReward(
            max_latency_ms=50.0,
            max_params_m=10.0,
            min_auroc=0.85,
        )

        metrics = {
            'auroc': 0.90,
            'latency_ms': 30.0,
            'params': 5e6,
        }

        reward, details = reward_fn.compute(metrics)

        assert details['meets_constraints'] is True
        assert reward > 0

    def test_auroc_below_threshold(self):
        """测试 AUROC 不达标"""
        reward_fn = EfficiencyConstraintReward(min_auroc=0.85)

        metrics = {'auroc': 0.80, 'latency_ms': 30, 'params': 5e6}

        reward, details = reward_fn.compute(metrics)

        assert details['meets_constraints'] is False
        assert 'AUROC' in details['reason']

    def test_latency_exceeds(self):
        """测试延迟超标"""
        reward_fn = EfficiencyConstraintReward(max_latency_ms=50.0)

        metrics = {'auroc': 0.90, 'latency_ms': 100.0, 'params': 5e6}

        reward, details = reward_fn.compute(metrics)

        assert details['meets_constraints'] is False
        assert 'Latency' in details['reason']


class TestSRLLMReward:
    """测试 SR-LLM 风格奖励"""

    def test_basic_computation(self):
        """测试基本计算"""
        reward_fn = SRLLMReward()

        metrics = {'auroc': 0.95}

        reward, scores = reward_fn.compute(metrics)

        assert 'fit_quality' in scores
        assert 'validity' in scores
        assert scores['validity'] == 1.0  # AUROC > 0.5

    def test_complexity_penalty(self):
        """测试复杂度惩罚"""
        reward_fn = SRLLMReward(complexity_weight=0.2)

        # 简单架构
        simple_arch = {'backbone': 'ResNet18', 'method': 'memory_bank', 'attention': 'none', 'levels': [2]}
        _, scores_simple = reward_fn.compute({'auroc': 0.9}, architecture=simple_arch)

        # 复杂架构
        complex_arch = {'backbone': 'ResNet50', 'method': 'student_teacher', 'attention': 'CBAM', 'levels': [2, 3, 4]}
        _, scores_complex = reward_fn.compute({'auroc': 0.9}, architecture=complex_arch)

        # 复杂架构的惩罚应该更高
        assert scores_complex['complexity_penalty'] > scores_simple['complexity_penalty']

    def test_novelty_bonus(self):
        """测试新颖性奖励"""
        reward_fn = SRLLMReward(novelty_bonus=0.1)

        history = [{'config': {'backbone': 'ResNet18'}}]

        # 与历史相同
        same_arch = {'backbone': 'ResNet18'}
        _, scores_same = reward_fn.compute({'auroc': 0.9}, architecture=same_arch, history=history)

        # 与历史不同
        diff_arch = {'backbone': 'MobileNetV3'}
        _, scores_diff = reward_fn.compute({'auroc': 0.9}, architecture=diff_arch, history=history)

        # 不同架构应该有更高新颖性
        assert scores_diff['novelty_bonus'] > scores_same['novelty_bonus']

    def test_invalid_architecture(self):
        """测试无效架构（AUROC < 0.5）"""
        reward_fn = SRLLMReward()

        metrics = {'auroc': 0.4}  # 无效

        reward, scores = reward_fn.compute(metrics)

        assert scores['validity'] == 0.0
        assert reward == 0.0


class TestComputeReward:
    """测试便捷函数"""

    def test_multi_dimension(self):
        """测试多维度策略"""
        reward, scores = compute_reward(
            {'auroc': 0.95, 'latency_ms': 30},
            strategy='multi_dimension',
        )
        assert reward > 0

    def test_efficiency_constraint(self):
        """测试效率约束策略"""
        reward, scores = compute_reward(
            {'auroc': 0.90, 'latency_ms': 30, 'params': 5e6},
            strategy='efficiency_constraint',
        )
        assert 'meets_constraints' in scores


class TestParetoFrontier:
    """测试帕累托前沿"""

    def test_get_pareto_frontier(self):
        """测试获取帕累托前沿"""
        reward_fn = MultiDimensionReward()

        results = [
            {'metrics': {'auroc': 0.95, 'latency_ms': 30, 'params': 10e6}},
            {'metrics': {'auroc': 0.90, 'latency_ms': 20, 'params': 10e6}},
            {'metrics': {'auroc': 0.95, 'latency_ms': 50, 'params': 5e6}},
        ]

        pareto = reward_fn.get_pareto_frontier(results)

        # 至少应该有一些非支配解
        assert len(pareto) >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
