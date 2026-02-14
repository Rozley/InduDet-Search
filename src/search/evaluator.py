"""
多保真度评估器
支持低保真→高保真逐步精化评估
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import roc_auc_score

from ..models.anomaly_detector import create_anomaly_detector
from ..data.datasets import MVTecSubset


class MultiFidelityEvaluator:
    """
    多保真度评估器

    策略:
    - Low: 1 epoch, 10%数据, ~30秒
    - Medium: 10 epochs, 50%数据, ~5分钟
    - High: 50 epochs, 100%数据, ~20分钟
    """

    def __init__(
        self,
        train_dataset,
        test_dataset,
        fidelity_levels: Optional[List[Dict]] = None,
        device: str = 'cuda',
    ):
        """
        Args:
            train_dataset: 训练数据集
            test_dataset: 测试数据集
            fidelity_levels: 保真度配置列表
            device: 计算设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        # 默认保真度配置
        self.fidelity_levels = fidelity_levels or [
            {
                'level': 'low',
                'epochs': 1,
                'data_ratio': 0.1,
                'threshold': 0.60,  # Low分数低于此值则停止
            },
            {
                'level': 'medium',
                'epochs': 10,
                'data_ratio': 0.5,
                'threshold': 0.75,  # Medium分数低于此值则停止
            },
            {
                'level': 'high',
                'epochs': 50,
                'data_ratio': 1.0,
                'threshold': 0.0,
            },
        ]

    def evaluate(
        self,
        config: Dict[str, Any],
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        评估架构

        Args:
            config: 架构配置
            verbose: 是否打印详细信息

        Returns:
            评估结果字典
        """
        results = {
            'config': config.copy(),
            'trial_log': [],
        }

        for level_config in self.fidelity_levels:
            level = level_config['level']
            epochs = level_config['epochs']
            data_ratio = level_config['data_ratio']
            threshold = level_config['threshold']

            if verbose:
                print(f"  [{level.upper()}] epochs={epochs}, data_ratio={data_ratio}")

            # 获取数据子集
            train_subset = MVTecSubset(self.train_dataset, ratio=data_ratio, shuffle=True)
            test_subset = MVTecSubset(self.test_dataset, ratio=data_ratio, shuffle=False)

            train_loader = DataLoader(
                train_subset,
                batch_size=32,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )
            test_loader = DataLoader(
                test_subset,
                batch_size=64,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )

            # 训练和评估
            level_metrics = self._train_and_eval(
                config=config,
                train_loader=train_loader,
                test_loader=test_loader,
                epochs=epochs,
                level=level,
                verbose=verbose,
            )

            results['trial_log'].append(level_metrics)

            # 检查是否应该停止
            if level_metrics['auroc'] < threshold:
                if verbose:
                    print(f"  [EARLY STOP] {level} AUROC {level_metrics['auroc']:.4f} < threshold {threshold}")

                results['final_auroc'] = level_metrics['auroc']
                results['final_level'] = level
                results['stopped_early'] = True
                results['latency_ms'] = level_metrics.get('latency_ms', 0)
                results['params'] = level_metrics.get('params', 0)

                return results

        # 所有级别都完成
        final_metrics = results['trial_log'][-1]
        results['final_auroc'] = final_metrics['auroc']
        results['final_level'] = 'high'
        results['stopped_early'] = False
        results['latency_ms'] = final_metrics.get('latency_ms', 0)
        results['params'] = final_metrics.get('params', 0)

        if verbose:
            print(f"  [FINAL] AUROC: {results['final_auroc']:.4f}")

        return results

    def _train_and_eval(
        self,
        config: Dict[str, Any],
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        level: str,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        训练并评估

        Args:
            config: 架构配置
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            epochs: 训练轮数
            level: 当前保真度级别

        Returns:
            评估指标
        """
        # 创建模型
        model = create_anomaly_detector(config)
        model = model.to(self.device)

        # 统计参数量
        params = model.count_parameters()

        # 训练（无监督，只有正常样本）
        if level in ['medium', 'high']:
            # 在medium/high级别，需要重新训练检测头
            model.fit(train_loader)

        # 评估
        all_scores = []
        all_labels = []

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                images = batch[0].to(self.device)
                labels = batch[1].numpy()

                output = model.predict(images)
                scores = output['image_score'].cpu().numpy()

                # 确保 scores 是 1D 数组
                if scores.ndim == 0:
                    scores = np.array([scores])

                all_scores.extend(scores)
                all_labels.extend(labels)

        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)

        # 计算AUROC
        try:
            auroc = roc_auc_score(all_labels, all_scores)
        except ValueError:
            # 只有一个类别
            auroc = 0.5 if all_labels.mean() < 0.5 else 1.0

        # 测量延迟
        latency_ms = self._measure_latency(model, test_loader)

        return {
            'level': level,
            'auroc': float(auroc),
            'params': params,
            'latency_ms': latency_ms,
        }

    def _measure_latency(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        warmup: int = 10,
        n_runs: int = 100,
    ) -> float:
        """测量推理延迟"""
        model.eval()

        # 获取一个batch
        batch = next(iter(test_loader))
        images = batch[0].to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(images)

        # 计时
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.time()

        with torch.no_grad():
            for _ in range(n_runs):
                _ = model(images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.time()

        # 计算平均延迟 (ms per image)
        total_images = images.shape[0] * n_runs
        total_time = (end_time - start_time) * 1000  # ms
        avg_latency = total_time / total_images

        return avg_latency

    def should_continue(self, current_metrics: Dict, fidelity: str) -> bool:
        """判断是否需要更高保真度评估"""
        if fidelity == 'low':
            return current_metrics['auroc'] >= self.fidelity_levels[0]['threshold']
        elif fidelity == 'medium':
            return current_metrics['auroc'] >= self.fidelity_levels[1]['threshold']
        return True


def create_evaluator(
    train_dataset,
    test_dataset,
    config: Optional[Dict] = None,
    device: str = 'cuda',
) -> MultiFidelityEvaluator:
    """创建评估器的便捷函数"""
    fidelity_levels = None
    if config:
        fidelity_levels = config.get('evaluator', {}).get('fidelity_levels')

    return MultiFidelityEvaluator(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        fidelity_levels=fidelity_levels,
        device=device,
    )
