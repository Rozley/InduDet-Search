"""
经验系统
简化版RAG - 使用JSON文件存储和检索案例
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class SimpleExperienceRAG:
    """
    简化版RAG经验系统

    功能:
    1. 存储成功/失败案例
    2. 检索相似成功案例
    3. 提供架构改进建议
    """

    def __init__(
        self,
        storage_path: str = './results/experiences.json',
        max_cases: int = 1000,
        success_threshold: float = 0.85,
        failure_threshold: float = 0.50,
    ):
        """
        Args:
            storage_path: 存储文件路径
            max_cases: 最大案例数量
            success_threshold: 成功阈值 (AUROC)
            failure_threshold: 失败阈值 (AUROC)
        """
        self.storage_path = storage_path
        self.max_cases = max_cases
        self.success_threshold = success_threshold
        self.failure_threshold = failure_threshold

        # 案例存储
        self.successful_cases: List[Dict] = []
        self.failed_cases: List[Dict] = []

        # 加载已有案例
        self._load()

    def add_case(
        self,
        config: Dict[str, Any],
        score: float,
        context: Optional[Dict] = None,
    ):
        """
        添加搜索案例

        Args:
            config: 架构配置
            score: AUROC分数
            context: 额外上下文信息
        """
        case = {
            'config': config.copy(),
            'score': score,
            'context': context or {},
            'timestamp': datetime.now().isoformat(),
        }

        if score >= self.success_threshold:
            # 高分案例
            self.successful_cases.append(case)
        elif score < self.failure_threshold:
            # 失败案例
            case['reason'] = self._analyze_failure_reason(config, score)
            self.failed_cases.append(case)

        # 保持最近max_cases条
        self._trim_cases()

        # 定期保存
        self._save()

    def retrieve_similar(
        self,
        current_config: Optional[Dict] = None,
        top_k: int = 3,
    ) -> List[Dict]:
        """
        检索相似的成功案例

        Args:
            current_config: 当前配置 (用于相似度匹配)
            top_k: 返回数量

        Returns:
            成功案例列表
        """
        if not self.successful_cases:
            return []

        # 按分数排序
        sorted_cases = sorted(
            self.successful_cases,
            key=lambda x: x['score'],
            reverse=True
        )

        return sorted_cases[:top_k]

    def get_design_suggestions(self, target_metrics: Optional[Dict] = None) -> List[str]:
        """
        基于历史经验生成设计建议

        Args:
            target_metrics: 目标指标

        Returns:
            建议列表
        """
        suggestions = []

        if not self.successful_cases:
            return ["建议从ResNet50 + memory_bank开始"]

        # 分析高分案例的模式
        backbone_counts = {}
        method_counts = {}
        level_counts = {}

        for case in self.successful_cases[-100:]:  # 最近100个
            config = case['config']

            # 统计backbone
            backbone = config.get('backbone', '')
            backbone_counts[backbone] = backbone_counts.get(backbone, 0) + 1

            # 统计method
            method = config.get('method', '')
            method_counts[method] = method_counts.get(method, 0) + 1

            # 统计levels
            levels = config.get('feature_levels', '')
            level_counts[levels] = level_counts.get(levels, 0) + 1

        # 生成建议
        if backbone_counts:
            best_backbone = max(backbone_counts, key=backbone_counts.get)
            suggestions.append(
                f"Backbone推荐: {best_backbone} "
                f"(成功率: {backbone_counts[best_backbone]}次)"
            )

        if method_counts:
            best_method = max(method_counts, key=method_counts.get)
            suggestions.append(
                f"Method推荐: {best_method} "
                f"(成功率: {method_counts[best_method]}次)"
            )

        if level_counts:
            best_levels = max(level_counts, key=level_counts.get)
            suggestions.append(f"特征层级推荐: {best_levels}")

        return suggestions

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'total_successful': len(self.successful_cases),
            'total_failed': len(self.failed_cases),
            'mean_score_successful': (
                sum(c['score'] for c in self.successful_cases) / len(self.successful_cases)
                if self.successful_cases else 0
            ),
            'best_score': (
                max(c['score'] for c in self.successful_cases)
                if self.successful_cases else 0
            ),
        }

    def _analyze_failure_reason(self, config: Dict, score: float) -> str:
        """分析失败原因"""
        if score < 0.5:
            return "检测性能过低(<0.5)，建议更换backbone或增加memory_size"
        elif score < 0.7:
            return "性能一般(0.5-0.7)，建议调整k值或尝试其他method"
        return "未知原因"

    def _trim_cases(self):
        """修剪案例列表，保持在max_cases以内"""
        total = len(self.successful_cases) + len(self.failed_cases)
        if total > self.max_cases:
            # 按时间保留最近的
            self.successful_cases = self.successful_cases[-self.max_cases//2:]
            self.failed_cases = self.failed_cases[-self.max_cases//2:]

    def _save(self):
        """保存到文件"""
        Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)

        data = {
            'successful_cases': self.successful_cases,
            'failed_cases': self.failed_cases,
            'last_saved': datetime.now().isoformat(),
        }

        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load(self):
        """从文件加载"""
        if not os.path.exists(self.storage_path):
            return

        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.successful_cases = data.get('successful_cases', [])
            self.failed_cases = data.get('failed_cases', [])

        except Exception as e:
            print(f"Warning: Failed to load experiences: {e}")

    def reset(self):
        """重置所有案例"""
        self.successful_cases.clear()
        self.failed_cases.clear()
        self._save()


def create_experience_rag(
    storage_path: str = './results/experiences.json',
    max_cases: int = 1000,
) -> SimpleExperienceRAG:
    """创建经验系统的便捷函数"""
    return SimpleExperienceRAG(
        storage_path=storage_path,
        max_cases=max_cases,
    )
