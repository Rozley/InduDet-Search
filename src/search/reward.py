"""
多维度奖励函数

参考 SR-LLM 设计，支持:
- 检测性能 (AUROC)
- 效率 (延迟、参数量)
- 泛化能力
- 复杂度惩罚
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class MultiDimensionReward:
    """
    多维度奖励函数

    综合考虑:
    - 检测性能 (AUROC, F1)
    - 推理效率 (延迟、参数量)
    - 泛化能力
    - 复杂度
    """

    # 默认权重配置
    DEFAULT_WEIGHTS = {
        'auroc': 0.40,        # 检测性能权重
        'f1': 0.10,           # F1 分数权重
        'latency': 0.20,     # 延迟惩罚权重
        'params': 0.15,       # 参数量惩罚权重
        'generalization': 0.10,  # 泛化能力权重
        'simplicity': 0.05,   # 简洁性奖励
    }

    # 效率基准值
    LATENCY_BASELINE_MS = 50.0    # 50ms 基准
    PARAMS_BASELINE_M = 10.0       # 10M 参数量基准

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        latency_threshold_ms: float = 100.0,
        params_threshold_m: float = 50.0,
    ):
        """
        Args:
            weights: 各维度权重
            latency_threshold_ms: 延迟阈值 (超过则惩罚)
            params_threshold_m: 参数量阈值 (超过则惩罚)
        """
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.latency_threshold = latency_threshold_ms
        self.params_threshold = params_threshold_m

    def compute(
        self,
        metrics: Dict[str, float],
        validation_metrics: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算综合奖励

        Args:
            metrics: 测试集指标
            validation_metrics: 验证集指标 (用于泛化计算)

        Returns:
            (综合奖励, 各维度分数)
        """
        scores = {}

        # 1. AUROC 分数 (0.5-1.0 -> 0-1)
        auroc = metrics.get('auroc', 0.0)
        scores['auroc'] = self._normalize_auroc(auroc)

        # 2. F1 分数 (0-1)
        f1 = metrics.get('f1', 0.0)
        scores['f1'] = max(0.0, f1)

        # 3. 延迟分数 (越小越好)
        latency = metrics.get('latency_ms', float('inf'))
        scores['latency'] = self._normalize_latency(latency)

        # 4. 参数量分数 (越小越好)
        params = metrics.get('params', float('inf'))
        scores['params'] = self._normalize_params(params)

        # 5. 泛化分数 (如果有验证集)
        if validation_metrics:
            test_auroc = metrics.get('auroc', 0.0)
            val_auroc = validation_metrics.get('auroc', 0.0)
            scores['generalization'] = self._normalize_generalization(test_auroc, val_auroc)
        else:
            scores['generalization'] = 0.5  # 默认中等

        # 6. 简洁性分数
        scores['simplicity'] = self._compute_simplicity(metrics)

        # 计算加权奖励
        reward = 0.0
        for key, weight in self.weights.items():
            reward += weight * scores.get(key, 0.0)

        return reward, scores

    def _normalize_auroc(self, auroc: float) -> float:
        """标准化 AUROC (0.5-1.0 -> 0-1)"""
        return max(0.0, min(1.0, (auroc - 0.5) * 2))

    def _normalize_latency(self, latency: float) -> float:
        """标准化延迟 (越小越好)"""
        if latency <= 0 or latency == float('inf'):
            return 0.0
        # 使用指数衰减
        return np.exp(-latency / self.LATENCY_BASELINE_MS)

    def _normalize_params(self, params: float) -> float:
        """标准化参数量 (越小越好)"""
        if params <= 0 or params == float('inf'):
            return 0.0
        # 使用对数变换
        return 1.0 / (1.0 + np.log1p(params / 1e6))

    def _normalize_generalization(self, test_auroc: float, val_auroc: float) -> float:
        """标准化泛化能力 (差异越小越好)"""
        if test_auroc <= 0 or val_auroc <= 0:
            return 0.5
        gap = abs(test_auroc - val_auroc)
        return max(0.0, 1.0 - gap * 2)

    def _compute_simplicity(self, metrics: Dict) -> float:
        """计算简洁性 (更简单的配置得高分)"""
        # 简单的启发式：层数越少越好
        levels = metrics.get('feature_levels', 3)
        simplicity = 1.0 / (1.0 + (levels - 1) * 0.2)
        return simplicity

    def get_pareto_frontier(
        self,
        results: List[Dict],
    ) -> List[Dict]:
        """
        获取帕累托前沿

        Args:
            results: 结果列表

        Returns:
            非支配解列表
        """
        pareto = []
        for r in results:
            dominated = False
            for other in results:
                if other != r and self._dominates(other, r):
                    dominated = True
                    break
            if not dominated:
                pareto.append(r)
        return pareto

    def _dominates(self, a: Dict, b: Dict) -> bool:
        """判断 a 是否支配 b"""
        # 使用 metrics 计算奖励
        reward_a, _ = self.compute(a.get('metrics', {}))
        reward_b, _ = self.compute(b.get('metrics', {}))

        # a 在所有维度都不比 b 差，且至少一个维度更好
        metrics_a = a.get('metrics', {})
        metrics_b = b.get('metrics', {})

        better_or_equal = (
            metrics_a.get('auroc', 0) >= metrics_b.get('auroc', 0) and
            metrics_a.get('latency_ms', float('inf')) <= metrics_b.get('latency_ms', float('inf')) and
            metrics_a.get('params', float('inf')) <= metrics_b.get('params', float('inf'))
        )

        strictly_better = (
            metrics_a.get('auroc', 0) > metrics_b.get('auroc', 0) or
            metrics_a.get('latency_ms', float('inf')) < metrics_b.get('latency_ms', float('inf')) or
            metrics_a.get('params', float('inf')) < metrics_b.get('params', float('inf'))
        )

        return better_or_equal and strictly_better


class EfficiencyConstraintReward:
    """
    效率约束奖励

    在满足效率约束的前提下最大化性能
    """

    def __init__(
        self,
        max_latency_ms: float = 50.0,
        max_params_m: float = 10.0,
        min_auroc: float = 0.85,
    ):
        """
        Args:
            max_latency_ms: 最大延迟
            max_params_m: 最大参数量
            min_auroc: 最小 AUROC
        """
        self.max_latency = max_latency_ms
        self.max_params = max_params_m * 1e6  # 转换为整数
        self.min_auroc = min_auroc

    def compute(
        self,
        metrics: Dict[str, float],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        计算奖励

        Returns:
            (奖励, 详细信息)
        """
        auroc = metrics.get('auroc', 0.0)
        latency = metrics.get('latency_ms', float('inf'))
        params = metrics.get('params', float('inf'))

        details = {
            'auroc': auroc,
            'latency_ms': latency,
            'params': params,
            'meets_constraints': True,
        }

        # 检查约束
        if auroc < self.min_auroc:
            # 性能不达标，惩罚
            reward = -1.0 + (auroc - self.min_auroc) * 10
            details['meets_constraints'] = False
            details['reason'] = f"AUROC {auroc:.4f} below threshold {self.min_auroc}"
            return reward, details

        if latency > self.max_latency:
            reward = -0.5 + (self.max_latency / latency)
            details['meets_constraints'] = False
            details['reason'] = f"Latency {latency:.2f}ms exceeds {self.max_latency}ms"
            return reward, details

        if params > self.max_params:
            reward = -0.5 + (self.max_params / params)
            details['meets_constraints'] = False
            details['reason'] = f"Params {params/1e6:.2f}M exceeds {self.max_params/1e6:.2f}M"
            return reward, details

        # 满足约束，给出性能奖励
        # 使用 log 奖励鼓励更高性能
        performance_reward = np.log1p(auroc - self.min_auroc) / np.log1p(1.0 - self.min_auroc)

        # 效率奖励
        latency_bonus = 1.0 - (latency / self.max_latency)
        params_bonus = 1.0 - (params / self.max_params)

        reward = performance_reward * 0.7 + latency_bonus * 0.15 + params_bonus * 0.15

        details['reason'] = "Meets all constraints"

        return reward, details


def compute_reward(
    metrics: Dict[str, float],
    strategy: str = 'multi_dimension',
    **kwargs,
) -> Tuple[float, Dict[str, float]]:
    """
    计算奖励的便捷函数

    Args:
        metrics: 性能指标
        strategy: 策略 ('multi_dimension', 'efficiency_constraint')
        **kwargs: 其他参数

    Returns:
        (奖励, 详细信息)
    """
    if strategy == 'multi_dimension':
        reward_fn = MultiDimensionReward(**kwargs)
    elif strategy == 'efficiency_constraint':
        reward_fn = EfficiencyConstraintReward(**kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return reward_fn.compute(metrics)


# ==================== SR-LLM 风格奖励函数 ====================

class SRLLMReward:
    """
    参考 SR-LLM 的奖励函数设计

    包含:
    - fit_quality: 拟合质量 (R2/AUROC)
    - complexity: 复杂度惩罚
    - novelty: 新颖性奖励
    - validity: 有效性检查
    """

    def __init__(
        self,
        complexity_weight: float = 0.1,
        novelty_bonus: float = 0.05,
    ):
        self.complexity_weight = complexity_weight
        self.novelty_bonus = novelty_bonus

    def compute(
        self,
        metrics: Dict[str, float],
        architecture: Optional[Dict] = None,
        history: Optional[List[Dict]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算奖励

        Args:
            metrics: 性能指标
            architecture: 架构配置 (用于计算复杂度)
            history: 历史结果 (用于计算新颖性)

        Returns:
            (奖励, 各维度分数)
        """
        scores = {}

        # 1. 拟合质量
        auroc = metrics.get('auroc', 0.0)
        scores['fit_quality'] = auroc  # AUROC 直接作为拟合质量

        # 2. 复杂度惩罚
        if architecture:
            complexity = self._compute_complexity(architecture)
            scores['complexity_penalty'] = complexity * self.complexity_weight
        else:
            scores['complexity_penalty'] = 0.0

        # 3. 新颖性奖励
        if history and architecture:
            novelty = self._compute_novelty(architecture, history)
            scores['novelty_bonus'] = novelty * self.novelty_bonus
        else:
            scores['novelty_bonus'] = 0.0

        # 4. 有效性检查
        scores['validity'] = 1.0 if auroc >= 0.5 else 0.0

        # 综合奖励
        reward = (
            scores['fit_quality'] +
            scores['novelty_bonus'] -
            scores['complexity_penalty']
        ) * scores['validity']

        return reward, scores

    def _compute_complexity(self, architecture: Dict) -> float:
        """计算架构复杂度"""
        complexity = 0.0

        # 层级复杂度
        levels = architecture.get('levels', [2, 3])
        complexity += len(levels) * 0.1

        # 注意力模块
        if architecture.get('attention') != 'none':
            complexity += 0.1

        # 方法复杂度
        method = architecture.get('method', '')
        if method == 'student_teacher':
            complexity += 0.2
        elif method == 'memory_bank':
            complexity += 0.1

        return complexity

    def _compute_novelty(
        self,
        architecture: Dict,
        history: List[Dict],
    ) -> float:
        """计算新颖性 (与历史结果的差异)"""
        if not history:
            return 1.0

        # 计算与最相似历史的差异
        max_similarity = 0.0
        for h in history:
            h_arch = h.get('config', {})
            similarity = self._architecture_similarity(architecture, h_arch)
            max_similarity = max(max_similarity, similarity)

        # 新颖性 = 1 - 相似度
        return 1.0 - max_similarity

    def _architecture_similarity(self, a: Dict, b: Dict) -> float:
        """计算两个架构的相似度"""
        if not a or not b:
            return 0.0

        similarity = 0.0
        count = 0

        for key in ['backbone', 'method', 'attention', 'levels']:
            if a.get(key) == b.get(key):
                similarity += 1.0
            count += 1

        return similarity / count if count > 0 else 0.0
