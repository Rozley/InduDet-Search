"""
搜索流程集成模块
整合 RAG、知识库、性能预测器到搜索流程
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..llm.rag import RAGSystem, ArchitectureKnowledgeBase
from .predictor import PerformancePredictor, EarlyStopPredictor, ArchitectureFeatureExtractor
from ..data.augmentation import AnomalyAugmentor, CutPaste


class SearchIntegrator:
    """
    搜索流程集成器

    整合以下组件:
    - RAG 知识库检索
    - 性能预测器
    - 早停机制
    - 数据增强
    """

    def __init__(
        self,
        config: Dict,
        use_rag: bool = True,
        use_predictor: bool = True,
        use_augmentation: bool = False,
        knowledge_path: str = './knowledge/vector_store.pkl',
    ):
        """
        Args:
            config: 配置字典
            use_rag: 是否使用 RAG
            use_predictor: 是否使用性能预测器
            use_augmentation: 是否使用数据增强
            knowledge_path: 知识库路径
        """
        self.config = config
        self.use_rag = use_rag
        self.use_predictor = use_predictor
        self.use_augmentation = use_augmentation

        # 初始化组件
        self.rag_system: Optional[RAGSystem] = None
        self.predictor: Optional[PerformancePredictor] = None
        self.early_stopper: Optional[EarlyStopPredictor] = None
        self.augmentor: Optional[AnomalyAugmentor] = None

        self._init_components(knowledge_path)

    def _init_components(self, knowledge_path: str):
        """初始化各组件"""
        # RAG 知识库
        if self.use_rag:
            try:
                kb = ArchitectureKnowledgeBase(
                    storage_path=knowledge_path,
                    use_default_knowledge=True,
                )
                self.rag_system = RAGSystem(knowledge_base=kb)
                print(f"RAG system initialized with {kb.get_statistics()['total_documents']} documents")
            except Exception as e:
                print(f"Warning: Failed to initialize RAG system: {e}")
                self.use_rag = False

        # 性能预测器
        if self.use_predictor:
            try:
                extractor = ArchitectureFeatureExtractor()
                self.predictor = PerformancePredictor(feature_extractor=extractor)
                self.early_stopper = EarlyStopPredictor(self.predictor)
                print("Performance predictor initialized")
            except Exception as e:
                print(f"Warning: Failed to initialize predictor: {e}")
                self.use_predictor = False

        # 数据增强
        if self.use_augmentation:
            try:
                self.augmentor = AnomalyAugmentor(
                    methods=['cutpaste', 'noise', 'blur'],
                    probabilities=[0.5, 0.3, 0.2],
                )
                print("Data augmentation initialized")
            except Exception as e:
                print(f"Warning: Failed to initialize augmentor: {e}")
                self.use_augmentation = False

    def get_rag_suggestion(
        self,
        task_spec: Dict,
    ) -> Dict:
        """
        获取 RAG 架构建议

        Args:
            task_spec: 任务规格

        Returns:
            建议结果
        """
        if self.rag_system is None:
            return {
                'suggested_config': {},
                'reasoning': 'RAG not available',
                'confidence': 0.0,
            }

        return self.rag_system.suggest_architecture(task_spec)

    def get_predicted_performance(
        self,
        architecture: Dict,
    ) -> Dict:
        """
        预测架构性能

        Args:
            architecture: 架构配置

        Returns:
            预测结果
        """
        if self.predictor is None:
            return {
                'predicted_auroc': 0.5,
                'estimated_latency_ms': 0.0,
            }

        return self.predictor.predict(architecture)

    def should_early_stop(
        self,
        architecture: Dict,
        current_fidelity: str,
        current_auroc: float,
    ) -> Tuple[bool, str]:
        """
        判断是否应该早停

        Args:
            architecture: 架构配置
            current_fidelity: 当前保真度
            current_auroc: 当前 AUROC

        Returns:
            (是否停止, 原因)
        """
        if self.early_stopper is None:
            return False, "Predictor not available"

        return self.early_stopper.should_stop(
            architecture, current_fidelity, current_auroc
        )

    def add_training_sample(
        self,
        architecture: Dict,
        metrics: Dict,
    ):
        """
        添加训练样本到预测器

        Args:
            architecture: 架构配置
            metrics: 性能指标
        """
        if self.predictor is not None:
            self.predictor.add_training_sample(architecture, metrics)

    def train_predictor(self):
        """训练性能预测器"""
        if self.predictor is not None:
            self.predictor.train()

    def get_knowledge_context(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[Dict]:
        """
        获取知识上下文

        Args:
            query: 查询
            top_k: 返回数量

        Returns:
            知识条目列表
        """
        if self.rag_system is None:
            return []

        return self.rag_system.retrieve(query, top_k)

    def get_status(self) -> Dict:
        """获取组件状态"""
        status = {
            'rag_enabled': self.use_rag and self.rag_system is not None,
            'predictor_enabled': self.use_predictor and self.predictor is not None,
            'augmentation_enabled': self.use_augmentation and self.augmentor is not None,
        }

        if self.rag_system:
            status['rag_stats'] = self.rag_system.knowledge_base.get_statistics()

        if self.predictor:
            status['predictor_trained'] = self.predictor.is_trained
            status['training_samples'] = len(self.predictor.training_data)

        return status


def create_search_integrator(
    config: Dict,
    knowledge_path: str = './knowledge/vector_store.pkl',
    use_rag: bool = True,
    use_predictor: bool = True,
    use_augmentation: bool = False,
) -> SearchIntegrator:
    """
    创建搜索集成器的便捷函数

    Args:
        config: 配置字典
        knowledge_path: 知识库路径
        use_rag: 是否使用 RAG
        use_predictor: 是否使用预测器
        use_augmentation: 是否使用增强

    Returns:
        SearchIntegrator 实例
    """
    return SearchIntegrator(
        config=config,
        use_rag=use_rag,
        use_predictor=use_predictor,
        use_augmentation=use_augmentation,
        knowledge_path=knowledge_path,
    )


class EnhancedArchitectureSearcher:
    """
    增强型架构搜索器

    在基础搜索器上集成 RAG、预测器等功能
    """

    def __init__(
        self,
        base_searcher,
        integrator: Optional[SearchIntegrator] = None,
    ):
        """
        Args:
            base_searcher: 基础搜索器
            integrator: 搜索集成器
        """
        self.base_searcher = base_searcher
        self.integrator = integrator or SearchIntegrator({})

    def run_with_integrator(
        self,
        n_trials: int = 200,
        verbose: bool = True,
        early_stop_enabled: bool = True,
    ) -> Tuple[List[Dict], Dict, float]:
        """
        使用集成器运行搜索

        Args:
            n_trials: 搜索次数
            verbose: 是否打印详细信息
            early_stop_enabled: 是否启用早停

        Returns:
            (results, best_config, best_score)
        """
        results = []
        best_config = None
        best_score = 0.0

        for trial in range(n_trials):
            # 获取 RAG 建议
            rag_suggestion = self.integrator.get_rag_suggestion({
                'category': getattr(self.base_searcher, 'category', 'unknown'),
                'trial': trial,
            })

            # 获取预测
            # ... (搜索逻辑)

            # 记录结果
            if self.integrator.predictor:
                self.integrator.add_training_sample(config, metrics)

            # 定期训练预测器
            if len(self.integrator.predictor.training_data) >= 20:
                self.integrator.train_predictor()

        return results, best_config, best_score

    def get_status(self) -> Dict:
        """获取搜索器状态"""
        return self.integrator.get_status()
