"""
主搜索流程
整合所有模块，执行完整的架构搜索
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from tqdm import tqdm

from ..utils.config import load_config
from ..search.search_space import get_search_space, SearchSpace
from ..search.searcher import create_searcher, IncrementalSearcher
from ..search.evaluator import create_evaluator, MultiFidelityEvaluator
from ..search.experience import create_experience_rag, SimpleExperienceRAG
from ..llm.agent import create_llm_agent, SimpleLLMAgent
from ..data.datasets import create_mvtec_datamodule, MVTecDataModule


class ArchitectureSearcher:
    """
    主架构搜索器

    整合:
    - 搜索策略 (随机+贝叶斯)
    - 多保真度评估器
    - 经验系统
    - LLM Agent
    """

    def __init__(
        self,
        config: Dict,
        train_dataset,
        test_dataset,
    ):
        """
        Args:
            config: 配置字典
            train_dataset: 训练数据集
            test_dataset: 测试数据集
        """
        self.config = config
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        # 提取配置
        search_strategy_cfg = config.get('search_strategy', {})
        evaluator_cfg = config.get('evaluator', {})
        experience_cfg = config.get('experience', {})
        llm_cfg = config.get('llm', {})
        output_cfg = config.get('output', {})
        resources_cfg = config.get('resources', {})

        # 初始化搜索器
        self.searcher = IncrementalSearcher(
            search_space=config.get('search_space'),
            n_random=search_strategy_cfg.get('n_random', 50),
            n_candidates=search_strategy_cfg.get('n_candidates', 100),
        )

        # 初始化评估器
        self.evaluator = MultiFidelityEvaluator(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            fidelity_levels=evaluator_cfg.get('fidelity_levels'),
            device=config.get('device', {}).get('type', 'cuda'),
        )

        # 初始化经验系统
        self.experience = SimpleExperienceRAG(
            storage_path=experience_cfg.get('storage_path', './results/experiences.json'),
            max_cases=experience_cfg.get('max_cases', 1000),
            success_threshold=experience_cfg.get('success_threshold', 0.85),
            failure_threshold=experience_cfg.get('failure_threshold', 0.50),
        )

        # 初始化LLM Agent
        self.llm_agent: Optional[SimpleLLMAgent] = None
        if llm_cfg.get('use_for_suggestion', True):
            self.llm_agent = create_llm_agent(
                use_for_suggestion=llm_cfg.get('use_for_suggestion', True),
            )

        # 输出配置
        self.save_dir = Path(output_cfg.get('save_dir', './results'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = output_cfg.get('checkpoint_interval', 20)
        self.log_interval = output_cfg.get('log_interval', 10)

        # 资源限制
        self.max_time_hours = resources_cfg.get('max_time_hours', 24)

        # 搜索状态
        self.results: List[Dict] = []
        self.best_config: Optional[Dict] = None
        self.best_score: float = 0.0
        self.start_time: float = 0.0

    def run(
        self,
        n_trials: int = 200,
        verbose: bool = True,
    ) -> Tuple[List[Dict], Dict, float]:
        """
        执行搜索

        Args:
            n_trials: 搜索次数
            verbose: 是否打印详细信息

        Returns:
            (results, best_config, best_score)
        """
        self.start_time = time.time()
        self.results = []

        print(f"\n{'='*60}")
        print(f"InduDet-Search 开始架构搜索")
        print(f"{'='*60}")
        print(f"搜索次数: {n_trials}")
        print(f"随机探索: {self.searcher.n_random}次")
        print(f"保真度级别: Low -> Medium -> High")
        print(f"{'='*60}\n")

        # 创建进度条
        pbar = tqdm(range(n_trials), desc="搜索进度")

        for trial in pbar:
            # 检查时间限制
            elapsed = (time.time() - self.start_time) / 3600
            if elapsed >= self.max_time_hours:
                print(f"\n时间限制达到 ({elapsed:.1f}小时)，停止搜索")
                break

            # 1. 采样架构
            if trial == 0 and self.llm_agent:
                # 第一个使用LLM建议
                history_summary = self.searcher.get_statistics()
                suggestion = self.llm_agent.suggest_architecture(
                    task_spec={'dataset': 'MVTec'},
                    history_summary=history_summary,
                )
                config = suggestion['config']
                if verbose:
                    print(f"\n[LLM建议] {suggestion.get('reasoning', '基于LLM分析')}")
            else:
                config = self.searcher.sample_architecture(trial)

            # 2. 评估架构
            if verbose:
                pbar.set_postfix({
                    'config': f"{config['backbone'][:8]}/{config['method'][:4]}",
                })

            metrics = self.evaluator.evaluate(config, verbose=verbose)

            # 3. 记录结果
            trial_result = {
                'trial': trial + 1,
                'config': config,
                'auroc': metrics['final_auroc'],
                'latency_ms': metrics.get('latency_ms', 0),
                'params': metrics.get('params', 0),
                'elapsed_hours': elapsed,
                'level': metrics.get('final_level', 'unknown'),
                'stopped_early': metrics.get('stopped_early', False),
            }
            self.results.append(trial_result)

            # 4. 更新组件
            self.searcher.update(config, metrics['final_auroc'])
            self.experience.add_case(config, metrics['final_auroc'])

            # 5. 更新最佳
            if metrics['final_auroc'] > self.best_score:
                self.best_score = metrics['final_auroc']
                self.best_config = config.copy()
                if verbose:
                    print(f"\n  [NEW BEST] AUROC: {self.best_score:.4f}")

            # 6. 定期保存
            if (trial + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint()

            # 7. 日志输出
            if (trial + 1) % self.log_interval == 0 and verbose:
                stats = self.searcher.get_statistics()
                print(f"\n[Trial {trial+1}] "
                      f"Best: {self.best_score:.4f} | "
                      f"Mean: {stats['mean_score']:.4f} | "
                      f"Random: {stats['n_random_phase']} | "
                      f"Bayesian: {stats['n_bayesian_phase']}")

        # 最终保存
        self._save_results()

        # 打印总结
        self._print_summary()

        return self.results, self.best_config, self.best_score

    def _save_checkpoint(self):
        """保存检查点"""
        checkpoint = {
            'results': self.results,
            'best_config': self.best_config,
            'best_score': self.best_score,
            'searcher_state': {
                'history': self.searcher.history,
                'X_train': [x.tolist() for x in self.searcher.X_train],
                'y_train': self.searcher.y_train,
            },
        }

        checkpoint_path = self.save_dir / 'checkpoint.json'
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        print(f"Checkpoint saved to {checkpoint_path}")

    def _save_results(self):
        """保存最终结果"""
        # 保存完整结果
        results_path = self.save_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        # 保存最佳配置
        if self.best_config:
            best_path = self.save_dir / 'best_config.json'
            with open(best_path, 'w') as f:
                json.dump({
                    'config': self.best_config,
                    'score': self.best_score,
                    'timestamp': datetime.now().isoformat(),
                }, f, indent=2)

        # 保存经验系统
        experience_path = self.save_dir / 'experiences.json'
        self.experience._save()

        print(f"\nResults saved to {self.save_dir}")

    def _print_summary(self):
        """打印总结"""
        total_time = (time.time() - self.start_time) / 3600

        print(f"\n{'='*60}")
        print(f"搜索完成!")
        print(f"{'='*60}")
        print(f"总搜索次数: {len(self.results)}")
        print(f"总耗时: {total_time:.2f}小时")
        print(f"最佳AUROC: {self.best_score:.4f}")
        print(f"\n最佳配置:")
        if self.best_config:
            for key, value in self.best_config.items():
                print(f"  {key}: {value}")
        print(f"{'='*60}")

    def get_pareto_frontier(self) -> List[Dict]:
        """获取帕累托最优解"""
        pareto = []
        for result in self.results:
            dominated = False
            for other in self.results:
                if other != result:
                    # 检查是否被支配 (更高的AUROC AND 更低的延迟)
                    if (other['auroc'] >= result['auroc'] and
                        other['latency_ms'] <= result['latency_ms'] and
                        (other['auroc'] > result['auroc'] or
                         other['latency_ms'] < result['latency_ms'])):
                        dominated = True
                        break
            if not dominated:
                pareto.append(result)
        return pareto


def run_search(
    config_path: str = None,
    data_dir: str = './data/mvtec',
    category: str = 'bottle',
    n_trials: int = 200,
    image_size: int = 224,
    batch_size: int = 32,
    save_dir: str = './results',
    device: str = 'cuda',
    verbose: bool = True,
) -> Tuple[List[Dict], Dict, float]:
    """
    执行架构搜索的便捷函数

    Args:
        config_path: 配置文件路径
        data_dir: 数据集目录
        category: 物体类别
        n_trials: 搜索次数
        image_size: 图像大小
        batch_size: 批次大小
        save_dir: 结果保存目录
        device: 计算设备
        verbose: 是否打印详细信息

    Returns:
        (results, best_config, best_score)
    """
    # 加载配置
    if config_path:
        config = load_config(config_path).to_dict()
    else:
        config = {}

    # 更新配置
    config['dataset'] = {
        'data_dir': data_dir,
        'image_size': image_size,
        'batch_size': batch_size,
    }
    config['output'] = {'save_dir': save_dir}
    config['device'] = {'type': device}

    # 创建数据模块
    datamodule = create_mvtec_datamodule(
        data_dir=data_dir,
        category=category,
        image_size=image_size,
        batch_size=batch_size,
    )

    # 创建搜索器
    searcher = ArchitectureSearcher(
        config=config,
        train_dataset=datamodule.get_train_dataset(),
        test_dataset=datamodule.get_test_dataset(),
    )

    # 执行搜索
    return searcher.run(n_trials=n_trials, verbose=verbose)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='InduDet-Search')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--data-dir', type=str, default='./data/mvtec')
    parser.add_argument('--category', type=str, default='bottle')
    parser.add_argument('--n-trials', type=int, default=200)
    parser.add_argument('--save-dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    results, best_config, best_score = run_search(
        config_path=args.config,
        data_dir=args.data_dir,
        category=args.category,
        n_trials=args.n_trials,
        save_dir=args.save_dir,
        device=args.device,
        verbose=args.verbose,
    )
