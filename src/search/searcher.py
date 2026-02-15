"""
增量式架构搜索器
简化版: 随机搜索 + 贝叶斯优化
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from .search_space import (
    SEARCH_SPACE,
    SearchSpace,
    sample_random_config,
    config_to_vector,
)


class IncrementalSearcher:
    """
    增量式架构搜索器

    搜索策略:
    1. Phase 1: 随机探索 (前50次)
    2. Phase 2: 贝叶斯优化 (基于历史结果)
    3. Phase 3: 经验引导 (可选RAG检索)
    """

    def __init__(
        self,
        search_space: Optional[Dict] = None,
        n_random: int = 50,
        n_candidates: int = 100,
        use_rag: bool = False,
        rag_system=None,
    ):
        """
        Args:
            search_space: 搜索空间配置
            n_random: 随机探索次数
            n_candidates: 贝叶斯优化候选数
            use_rag: 是否使用RAG引导
            rag_system: RAG系统实例
        """
        self.search_space_obj = SearchSpace(search_space)
        self.n_random = n_random
        self.n_candidates = n_candidates
        self.use_rag = use_rag
        self.rag = rag_system

        # 历史记录
        self.history: List[Dict] = []

        # 贝叶斯优化模型
        self.gp_model: Optional[GaussianProcessRegressor] = None
        self.X_train: List[np.ndarray] = []
        self.y_train: List[float] = []

        # 随机种子
        self.seed = 42
        np.random.seed(self.seed)

    def sample_architecture(self, iteration: int) -> Dict[str, Any]:
        """
        采样一个架构配置

        Args:
            iteration: 当前迭代次数 (0-based)

        Returns:
            架构配置字典
        """
        if iteration < self.n_random:
            # Phase 1: 随机探索
            return self._sample_random()
        elif self.gp_model is None or len(self.history) < self.n_random + 10:
            # Phase 2a: 收集足够数据后再用贝叶斯
            return self._sample_random()
        else:
            # Phase 2b: 贝叶斯优化
            return self._sample_bayesian()

    def _sample_random(self) -> Dict[str, Any]:
        """随机采样"""
        return self.search_space_obj.sample()

    def _sample_bayesian(self) -> Dict[str, Any]:
        """贝叶斯优化采样"""
        # 1. 获取候选点
        candidates = self._generate_candidates(n=self.n_candidates)

        # 2. 预测每个候选的期望改善 (EI)
        X_candidates = self._encode_candidates(candidates)
        mu, sigma = self.gp_model.predict(X_candidates, return_std=True)

        # 3. 计算 Expected Improvement
        best_y = max(self.y_train)
        ei = self._compute_ei(mu, sigma, best_y)

        # 4. 选择EI最大的候选
        best_idx = np.argmax(ei)
        return candidates[best_idx]

    def _compute_ei(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        best_y: float,
        xi: float = 0.01,
    ) -> np.ndarray:
        """
        计算Expected Improvement

        Args:
            mu: 预测均值
            sigma: 预测标准差
            best_y: 当前最佳分数
            xi: 探索参数
        """
        with np.errstate(divide='warn', invalid='warn'):
            improvement = (mu - best_y)
            # 确保improvement是标量或数组
            if isinstance(improvement, (int, float)):
                improvement = np.array([improvement])

            Z = np.where(sigma > 0, (improvement - xi) / sigma, 0)
            ei = np.where(sigma > 0,
                         (improvement - xi) * self._norm_cdf(Z) + sigma * self._norm_pdf(Z),
                         0.0)

            # 确保非负
            ei = np.maximum(ei, 0)

        return ei

    def _norm_cdf(self, x: np.ndarray) -> np.ndarray:
        """标准正态分布CDF"""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x * x / np.pi)))

    def _norm_pdf(self, x: np.ndarray) -> np.ndarray:
        """标准正态分布PDF"""
        return np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)

    def _generate_candidates(self, n: int) -> List[Dict[str, Any]]:
        """生成候选配置"""
        candidates = []
        for _ in range(n):
            config = self.search_space_obj.sample()
            candidates.append(config)
        return candidates

    def _encode_candidates(self, candidates: List[Dict[str, Any]]) -> np.ndarray:
        """将配置列表编码为矩阵"""
        encoded = []
        for config in candidates:
            vector = self.search_space_obj.encode(config)
            encoded.append(vector)
        return np.array(encoded)

    def update(self, config: Dict[str, Any], score: float):
        """
        更新搜索模型

        Args:
            config: 架构配置
            score: 评估分数 (AUROC)
        """
        # 跳过 NaN 分数
        if np.isnan(score):
            print(f"[Warning] Skipping config with NaN score: {config}")
            return

        # 记录历史
        self.history.append({
            'config': config.copy(),
            'score': score,
        })

        # 更新训练数据
        X = self.search_space_obj.encode(config)
        self.X_train.append(X)
        self.y_train.append(score)

        # 重新训练GP
        if len(self.X_train) >= 10:
            self._fit_gp_model()

    def _fit_gp_model(self):
        """训练高斯过程模型"""
        X_train = np.array(self.X_train)
        y_train = np.array(self.y_train)

        # 过滤 NaN 值
        valid_mask = ~np.isnan(y_train)
        if not np.any(valid_mask):
            return  # 所有值都是 NaN，跳过拟合

        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]

        # 标准化y
        y_mean = y_train.mean()
        y_std = y_train.std()

        self.gp_model = GaussianProcessRegressor(
            kernel=Matern(nu=2.5, length_scale=1.0),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
        )
        self.gp_model.fit(X_train, y_train)

        # 保存标准化参数
        self.y_mean = y_mean
        self.y_std = max(y_std, 1e-6)

    def get_best_config(self) -> Optional[Dict[str, Any]]:
        """获取历史最佳配置"""
        if not self.history:
            return None

        best = max(self.history, key=lambda x: x['score'])
        return best['config'].copy()

    def get_best_score(self) -> float:
        """获取历史最佳分数"""
        if not self.history:
            return 0.0

        return max(h['score'] for h in self.history)

    def get_statistics(self) -> Dict:
        """获取搜索统计信息"""
        return {
            'total_trials': len(self.history),
            'best_score': self.get_best_score(),
            'mean_score': np.mean([h['score'] for h in self.history]) if self.history else 0.0,
            'std_score': np.std([h['score'] for h in self.history]) if self.history else 0.0,
            'n_random_phase': min(len(self.history), self.n_random),
            'n_bayesian_phase': max(0, len(self.history) - self.n_random),
            'has_gp_model': self.gp_model is not None,
        }

    def reset(self):
        """重置搜索器"""
        self.history.clear()
        self.X_train.clear()
        self.y_train.clear()
        self.gp_model = None
        np.random.seed(self.seed)


def create_searcher(
    search_space: Optional[Dict] = None,
    n_random: int = 50,
    n_total: int = 200,
    use_rag: bool = False,
    rag_system=None,
) -> IncrementalSearcher:
    """创建搜索器的便捷函数"""
    return IncrementalSearcher(
        search_space=search_space,
        n_random=n_random,
        n_candidates=n_total // 10,  # 候选数为总次数的1/10
        use_rag=use_rag,
        rag_system=rag_system,
    )
