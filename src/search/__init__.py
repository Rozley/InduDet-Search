"""
主搜索流程
整合所有模块，执行完整的架构搜索
支持单类别和多类别搜索
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from tqdm import tqdm


class NumpyEncoder(json.JSONEncoder):
    """JSON编码器，支持numpy类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


from ..utils.config import load_config
from ..search.search_space import get_search_space, SearchSpace
from ..search.searcher import create_searcher, IncrementalSearcher
from ..search.evaluator import create_evaluator, MultiFidelityEvaluator
from ..search.experience import create_experience_rag, SimpleExperienceRAG
from ..llm.agent import create_llm_agent, SimpleLLMAgent
from ..data.datasets import create_mvtec_datamodule, MVTecDataModule


class ArchitectureSearcher:
    """
    主架构搜索器（单类别版本）

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
        category: str = 'unknown',
    ):
        """
        Args:
            config: 配置字典
            train_dataset: 训练数据集
            test_dataset: 测试数据集
            category: 当前搜索的类别名称
        """
        self.config = config
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.category = category

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
        print(f"类别: {self.category}")
        print(f"搜索次数: {n_trials}")
        print(f"{'='*60}\n")

        pbar = tqdm(range(n_trials), desc=f"[{self.category}]")

        for trial in pbar:
            elapsed = (time.time() - self.start_time) / 3600
            if elapsed >= self.max_time_hours:
                print(f"\n时间限制达到，停止搜索")
                break

            # 采样架构
            if trial == 0 and self.llm_agent:
                suggestion = self.llm_agent.suggest_architecture(
                    task_spec={'dataset': 'MVTec', 'category': self.category},
                    history_summary=self.searcher.get_statistics(),
                )
                config = suggestion['config']
                if verbose:
                    print(f"\n[LLM建议] {suggestion.get('reasoning', '')}")
            else:
                config = self.searcher.sample_architecture(trial)

            if verbose:
                pbar.set_postfix({
                    'config': f"{config['backbone'][:8]}/{config['method'][:4]}",
                })

            # 评估架构
            metrics = self.evaluator.evaluate(config, verbose=False)

            # 记录结果
            trial_result = {
                'trial': trial + 1,
                'config': config,
                'auroc': metrics['final_auroc'],
                'latency_ms': metrics.get('latency_ms', 0),
                'params': metrics.get('params', 0),
                'elapsed_hours': elapsed,
                'level': metrics.get('final_level', 'unknown'),
                'stopped_early': metrics.get('stopped_early', False),
                'category': self.category,
            }
            self.results.append(trial_result)

            # 更新组件
            self.searcher.update(config, metrics['final_auroc'])
            self.experience.add_case(config, metrics['final_auroc'])

            # 更新最佳
            if metrics['final_auroc'] > self.best_score:
                self.best_score = metrics['final_auroc']
                self.best_config = config.copy()
                if verbose:
                    print(f"\n  [NEW BEST] AUROC: {self.best_score:.4f}")

            # 定期保存
            if (trial + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint()

        return self.results, self.best_config, self.best_score

    def _save_checkpoint(self):
        """保存检查点"""
        checkpoint = {
            'results': self.results,
            'best_config': self.best_config,
            'best_score': self.best_score,
            'category': self.category,
        }
        checkpoint_path = self.save_dir / f'checkpoint_{self.category}.json'
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2, cls=NumpyEncoder)

    def _save_results(self):
        """保存结果"""
        results_path = self.save_dir / f'results_{self.category}.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, cls=NumpyEncoder)


class MultiCategorySearcher:
    """
    多类别架构搜索器

    支持:
    - 顺序搜索: 每个类别独立搜索
    - 联合搜索: 所有类别共享搜索经验
    """

    def __init__(
        self,
        config: Dict,
        categories: List[str],
        data_dir: str = './data/mvtec',
        strategy: str = 'joint',
        **kwargs,
    ):
        """
        Args:
            config: 配置字典
            categories: 要搜索的类别列表
            data_dir: 数据集目录
            strategy: 搜索策略 ('sequential' 或 'joint')
        """
        self.config = config
        self.categories = categories
        self.data_dir = data_dir
        self.strategy = strategy

        # 提取配置
        self.image_size = config.get('dataset', {}).get('image_size', 224)
        self.batch_size = config.get('dataset', {}).get('batch_size', 32)
        self.output_dir = Path(config.get('output', {}).get('save_dir', './results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 搜索器缓存 (用于联合搜索)
        self.searchers: Dict[str, ArchitectureSearcher] = {}

        # 全局最佳配置
        self.global_best_config: Optional[Dict] = None
        self.global_best_score: float = 0.0

        # 所有类别的结果汇总
        self.all_results: Dict[str, List[Dict]] = {}
        self.category_best: Dict[str, Dict] = {}

    def run(
        self,
        n_trials: int = 200,
        verbose: bool = True,
    ) -> Tuple[Dict[str, Any], Dict, float]:
        """
        执行多类别搜索

        Args:
            n_trials: 每个类别的搜索次数
            verbose: 是否打印详细信息

        Returns:
            (summary, best_config, best_score)
        """
        print(f"\n{'='*60}")
        print(f"多类别架构搜索")
        print(f"类别数: {len(self.categories)}")
        print(f"策略: {self.strategy}")
        print(f"搜索次数/类别: {n_trials}")
        print(f"{'='*60}\n")

        total_start = time.time()

        if self.strategy == 'joint':
            results = self._run_joint_search(n_trials, verbose)
        else:
            results = self._run_sequential_search(n_trials, verbose)

        total_time = (time.time() - total_start) / 3600

        # 生成汇总报告
        summary = self._generate_summary(results, total_time)

        return summary, self.global_best_config, self.global_best_score

    def _run_sequential_search(
        self,
        n_trials: int,
        verbose: bool,
    ) -> Dict[str, List[Dict]]:
        """
        顺序搜索: 每个类别独立搜索
        """
        results = {}

        for i, category in enumerate(self.categories):
            print(f"\n{'='*40}")
            print(f"[{i+1}/{len(self.categories)}] 搜索类别: {category}")
            print(f"{'='*40}")

            # 创建数据模块
            datamodule = create_mvtec_datamodule(
                data_dir=self.data_dir,
                category=category,
                image_size=self.image_size,
                batch_size=self.batch_size,
            )

            # 创建搜索器
            searcher = ArchitectureSearcher(
                config=self.config,
                train_dataset=datamodule.get_train_dataset(),
                test_dataset=datamodule.get_test_dataset(),
                category=category,
            )

            # 执行搜索
            category_results, best_config, best_score = searcher.run(
                n_trials=n_trials,
                verbose=verbose,
            )

            results[category] = category_results
            self.category_best[category] = {
                'config': best_config,
                'score': best_score,
            }

            # 更新全局最佳
            if best_score > self.global_best_score:
                self.global_best_score = best_score
                self.global_best_config = best_config.copy()
                print(f"\n  [GLOBAL BEST] AUROC: {best_score:.4f}")

        return results

    def _run_joint_search(
        self,
        n_trials: int,
        verbose: bool,
    ) -> Dict[str, List[Dict]]:
        """
        联合搜索: 共享搜索经验

        策略:
        1. 在第一个类别上完整搜索
        2. 其他类别使用已有经验加速
        """
        results = {}

        # 第一类别: 完整搜索
        first_category = self.categories[0]
        print(f"\n{'='*40}")
        print(f"[1/{len(self.categories)}] 完整搜索: {first_category}")
        print(f"{'='*40}")

        datamodule = create_mvtec_datamodule(
            data_dir=self.data_dir,
            category=first_category,
            image_size=self.image_size,
            batch_size=self.batch_size,
        )

        first_searcher = ArchitectureSearcher(
            config=self.config,
            train_dataset=datamodule.get_train_dataset(),
            test_dataset=datamodule.get_test_dataset(),
            category=first_category,
        )

        first_results, first_best, first_score = first_searcher.run(
            n_trials=n_trials,
            verbose=verbose,
        )

        results[first_category] = first_results
        self.category_best[first_category] = {
            'config': first_best,
            'score': first_score,
        }

        self.global_best_config = first_best.copy()
        self.global_best_score = first_score

        # 其他类别: 使用经验加速
        for i, category in enumerate(self.categories[1:], 2):
            print(f"\n{'='*40}")
            print(f"[{i}/{len(self.categories)}] 加速搜索: {category}")
            print(f"使用已有搜索经验...")
            print(f"{'='*40}")

            datamodule = create_mvtec_datamodule(
                data_dir=self.data_dir,
                category=category,
                image_size=self.image_size,
                batch_size=self.batch_size,
            )

            # 减少搜索次数 (使用经验后可以减少)
            reduced_trials = max(50, n_trials // 2)

            searcher = ArchitectureSearcher(
                config=self.config,
                train_dataset=datamodule.get_train_dataset(),
                test_dataset=datamodule.get_test_dataset(),
                category=category,
            )

            # 使用第一个类别的经验初始化
            if first_searcher.searcher.history:
                for h in first_searcher.searcher.history:
                    searcher.searcher.update(h['config'], h['score'])

            category_results, best_config, best_score = searcher.run(
                n_trials=reduced_trials,
                verbose=verbose,
            )

            results[category] = category_results
            self.category_best[category] = {
                'config': best_config,
                'score': best_score,
            }

            if best_score > self.global_best_score:
                self.global_best_score = best_score
                self.global_best_config = best_config.copy()
                print(f"\n  [GLOBAL BEST] AUROC: {best_score:.4f}")

        return results

    def _generate_summary(
        self,
        results: Dict[str, List[Dict]],
        total_time: float,
    ) -> Dict[str, Any]:
        """生成汇总报告"""
        # 计算每个类别的平均AUROC
        category_scores = {}
        for category, category_results in results.items():
            if category_results:
                scores = [r['auroc'] for r in category_results]
                category_scores[category] = {
                    'mean_auroc': np.mean(scores),
                    'std_auroc': np.std(scores),
                    'max_auroc': np.max(scores),
                    'min_auroc': np.min(scores),
                    'n_trials': len(scores),
                }

        # 全局统计
        all_scores = [s for cs in category_scores.values() for s in [cs['mean_auroc']]]

        summary = {
            'categories': self.categories,
            'n_categories': len(self.categories),
            'strategy': self.strategy,
            'total_time_hours': total_time,
            'global_best_config': self.global_best_config,
            'global_best_score': self.global_best_score,
            'category_results': self.category_best,
            'category_statistics': category_scores,
            'mean_auroc': np.mean(all_scores) if all_scores else 0,
            'std_auroc': np.std(all_scores) if all_scores else 0,
        }

        # 保存汇总
        summary_path = self.output_dir / 'multi_category_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)

        # 打印汇总
        print(f"\n{'='*60}")
        print(f"多类别搜索汇总")
        print(f"{'='*60}")
        print(f"类别数: {len(self.categories)}")
        print(f"总耗时: {total_time:.2f}小时")
        print(f"全局最佳AUROC: {self.global_best_score:.4f}")
        print(f"平均AUROC: {summary['mean_auroc']:.4f} ± {summary['std_auroc']:.4f}")
        print(f"\n各类别最佳:")
        for cat, info in self.category_best.items():
            print(f"  {cat}: {info['score']:.4f}")
        print(f"{'='*60}")

        return summary


def aggregate_category_metrics(
    category_metrics: Dict[str, float]
) -> Dict[str, float]:
    """计算所有类别的聚合指标"""
    scores = list(category_metrics.values())
    return {
        'mean_auroc': np.mean(scores),
        'std_auroc': np.std(scores),
        'min_auroc': np.min(scores),
        'max_auroc': np.max(scores),
    }


def run_search(
    config_path: str = None,
    data_dir: str = './data/mvtec',
    categories: Union[str, List[str]] = 'bottle',
    n_trials: int = 200,
    image_size: int = 224,
    batch_size: int = 32,
    strategy: str = 'joint',
    save_dir: str = './results',
    device: str = 'cuda',
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Dict, float]:
    """
    执行架构搜索的便捷函数

    支持单类别和多类别搜索

    Args:
        config_path: 配置文件路径
        data_dir: 数据集目录
        categories: 类别列表或单个类别名称
        n_trials: 搜索次数
        image_size: 图像大小
        batch_size: 批次大小
        strategy: 搜索策略 ('sequential' 或 'joint')
        save_dir: 结果保存目录
        device: 计算设备
        verbose: 是否打印详细信息

    Returns:
        (summary, best_config, best_score)
    """
    # 解析类别
    if isinstance(categories, str):
        category_list = [categories]
    else:
        category_list = categories

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

    if len(category_list) == 1:
        # 单类别搜索
        category = category_list[0]
        datamodule = create_mvtec_datamodule(
            data_dir=data_dir,
            category=category,
            image_size=image_size,
            batch_size=batch_size,
        )

        searcher = ArchitectureSearcher(
            config=config,
            train_dataset=datamodule.get_train_dataset(),
            test_dataset=datamodule.get_test_dataset(),
            category=category,
        )

        results, best_config, best_score = searcher.run(
            n_trials=n_trials,
            verbose=verbose,
        )

        summary = {
            'categories': [category],
            'category_results': {category: {'config': best_config, 'score': best_score}},
            'global_best_config': best_config,
            'global_best_score': best_score,
        }

        return summary, best_config, best_score

    else:
        # 多类别搜索
        multi_searcher = MultiCategorySearcher(
            config=config,
            categories=category_list,
            data_dir=data_dir,
            strategy=strategy,
        )

        return multi_searcher.run(n_trials=n_trials, verbose=verbose)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='InduDet-Search')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--data-dir', type=str, default='./data/mvtec')
    parser.add_argument('--categories', type=str, default='bottle',
                        help='逗号分隔的类别列表')
    parser.add_argument('--all-categories', action='store_true',
                        help='使用所有MVTec AD类别')
    parser.add_argument('--n-trials', type=int, default=200)
    parser.add_argument('--strategy', type=str, default='joint',
                        choices=['sequential', 'joint'])
    parser.add_argument('--save-dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    # 解析类别
    if args.all_categories:
        categories = [
            'bottle', 'cable', 'capsule', 'carpet', 'grid',
            'guitar', 'hazelnut', 'leather', 'metal_nut', 'pill',
            'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
        ]
    else:
        categories = [c.strip() for c in args.categories.split(',')]

    summary, best_config, best_score = run_search(
        config_path=args.config,
        data_dir=args.data_dir,
        categories=categories,
        n_trials=args.n_trials,
        strategy=args.strategy,
        save_dir=args.save_dir,
        device=args.device,
        verbose=args.verbose,
    )
