"""
RAG 知识库系统
基于向量检索的架构知识管理系统
"""

import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimpleVectorStore:
    """
    简单的向量存储 (基于 TF-IDF)

    用于在没有外部向量数据库的情况下实现基础检索功能
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
        )
        self.vectors = None
        self.documents = []
        self.metadata = []

    def add(
        self,
        documents: List[str],
        metadata: Optional[List[Dict]] = None,
    ):
        """
        添加文档

        Args:
            documents: 文档内容列表
            metadata: 元数据列表
        """
        # 向量化
        self.vectors = self.vectorizer.fit_transform(documents)

        # 存储
        self.documents.extend(documents)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(documents))

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        搜索

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            结果列表
        """
        if self.vectors is None or len(self.documents) == 0:
            return []

        # 向量化查询
        query_vector = self.vectorizer.transform([query])

        # 计算相似度
        similarities = cosine_similarity(query_vector, self.vectors)[0]

        # 排序
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'metadata': self.metadata[idx],
                'score': float(similarities[idx]),
            })

        return results

    def save(self, path: str):
        """保存到文件"""
        data = {
            'vectorizer': self.vectorizer,
            'vectors': self.vectors,
            'documents': self.documents,
            'metadata': self.metadata,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str):
        """从文件加载"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.vectorizer = data['vectorizer']
        self.vectors = data['vectors']
        self.documents = data['documents']
        self.metadata = data['metadata']


class ArchitectureKnowledgeBase:
    """
    架构知识库

    存储和管理架构设计相关的知识，包括：
    - 经典论文方法
    - 历史搜索经验
    - 成功/失败模式
    """

    # 默认知识模板
    DEFAULT_KNOWLEDGE = [
        {
            'content': "PatchCore uses WideResNet50 as backbone with memory bank. Features are extracted from intermediate layers 2 and 3. K-nearest neighbors with k=9 for anomaly scoring. Achieves 99.6% image-level AUROC on MVTec AD.",
            'category': 'method',
            'source': 'PatchCore (CVPR 2022)',
            'performance': {'auroc': 0.996, 'method': 'memory_bank'}
        },
        {
            'content': "PaDiM uses ResNet50 with distribution-based detection. Features from layers 1, 2, 3 are concatenated. PCA reduces dimensionality to 100. Mahalanobis distance for anomaly scoring. Achieves 97.8% image-level AUROC.",
            'category': 'method',
            'source': 'PaDiM (CVPR 2021)',
            'performance': {'auroc': 0.978, 'method': 'distribution'}
        },
        {
            'content': "WideResNet50 provides better feature representation than ResNet18 for industrial anomaly detection. Deeper networks capture more semantic information useful for detecting complex anomalies.",
            'category': 'insight',
            'source': 'empirical_study',
            'performance': {'backbone': 'WideResNet50'}
        },
        {
            'content': "CBAM attention module improves detection accuracy by focusing on informative regions. Adding CBAM to ResNet18 increases AUROC by 1-3% on average.",
            'category': 'insight',
            'source': 'empirical_study',
            'performance': {'attention': 'CBAM', 'improvement': 0.02}
        },
        {
            'content': "Memory bank size of 1000 provides good balance between representation and speed. Larger banks (5000+) give marginal improvement but increase inference time.",
            'category': 'insight',
            'source': 'empirical_study',
            'performance': {'memory_bank_size': 1000}
        },
        {
            'content': "K-center sampling is preferred over random sampling for memory bank construction. It ensures better coverage of the feature space.",
            'category': 'insight',
            'source': 'empirical_study',
            'performance': {'sampling': 'kcenter'}
        },
        {
            'content': "Multi-scale features from different backbone layers improve anomaly localization. Combining layer 2 and layer 3 features achieves better results than single layer.",
            'category': 'insight',
            'source': 'empirical_study',
            'performance': {'feature_levels': [2, 3]}
        },
        {
            'content': "EfficientNet-B0 is efficient for edge deployment with competitive accuracy. Achieves 92-95% AUROC with 5M parameters.",
            'category': 'insight',
            'source': 'empirical_study',
            'performance': {'backbone': 'EfficientNet-B0', 'params': 5e6}
        },
    ]

    def __init__(
        self,
        storage_path: str = './knowledge/vector_store.pkl',
        use_default_knowledge: bool = True,
    ):
        """
        Args:
            storage_path: 存储路径
            use_default_knowledge: 是否加载默认知识
        """
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        self.store = SimpleVectorStore()

        # 加载默认知识或已有知识
        if use_default_knowledge:
            if self.storage_path.exists():
                self.store.load(str(self.storage_path))
            else:
                self._load_default_knowledge()
                self.save()

    def _load_default_knowledge(self):
        """加载默认知识"""
        documents = [k['content'] for k in self.DEFAULT_KNOWLEDGE]
        metadata = self.DEFAULT_KNOWLEDGE
        self.store.add(documents, metadata)

    def search(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        搜索知识

        Args:
            query: 查询文本
            top_k: 返回数量
            category_filter: 类别过滤

        Returns:
            知识条目列表
        """
        results = self.store.search(query, top_k)

        # 过滤
        if category_filter:
            results = [
                r for r in results
                if r['metadata'].get('category') == category_filter
            ]

        return results

    def add_knowledge(
        self,
        content: str,
        category: str = 'experience',
        source: str = 'search',
        performance: Optional[Dict] = None,
        **metadata,
    ):
        """
        添加新知识

        Args:
            content: 知识内容
            category: 类别 (method, insight, experience)
            source: 来源
            performance: 性能数据
            **metadata: 其他元数据
        """
        knowledge = {
            'content': content,
            'category': category,
            'source': source,
            'performance': performance or {},
            'timestamp': datetime.now().isoformat(),
            **metadata,
        }

        self.store.add([content], [knowledge])
        self.save()

    def add_architecture_experience(
        self,
        architecture: Dict,
        metrics: Dict,
        is_success: bool = True,
    ):
        """
        添加架构搜索经验

        Args:
            architecture: 架构配置
            metrics: 性能指标
            is_success: 是否成功
        """
        # 构建描述
        backbone = architecture.get('backbone', 'unknown')
        method = architecture.get('method', 'unknown')
        attention = architecture.get('attention', 'none')

        if is_success:
            content = (
                f"Successful architecture: {backbone} backbone with {method} detection method "
                f"and {attention} attention. Achieved AUROC: {metrics.get('auroc', 0):.4f}, "
                f"latency: {metrics.get('latency_ms', 0):.2f}ms, "
                f"parameters: {metrics.get('params', 0):,}"
            )
        else:
            content = (
                f"Failed architecture: {backbone} backbone with {method} detection method "
                f"and {attention} attention. AUROC: {metrics.get('auroc', 0):.4f}. "
                f"Consider alternative configurations."
            )

        self.add_knowledge(
            content=content,
            category='experience',
            source='search',
            performance={
                'auroc': metrics.get('auroc', 0),
                'latency_ms': metrics.get('latency_ms', 0),
                'params': metrics.get('params', 0),
                'is_success': is_success,
            },
            architecture=architecture,
        )

    def get_successful_patterns(self) -> List[Dict]:
        """获取成功模式"""
        results = self.store.search("successful architecture", top_k=10)
        return [
            r for r in results
            if r['metadata'].get('performance', {}).get('is_success', False)
        ]

    def get_failed_patterns(self) -> List[Dict]:
        """获取失败模式"""
        results = self.store.search("failed architecture", top_k=10)
        return [
            r for r in results
            if not r['metadata'].get('performance', {}).get('is_success', True)
        ]

    def save(self):
        """保存知识库"""
        self.store.save(str(self.storage_path))

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'total_documents': len(self.store.documents),
            'categories': list(set(
                m.get('category', 'unknown')
                for m in self.store.metadata
            )),
            'sources': list(set(
                m.get('source', 'unknown')
                for m in self.store.metadata
            )),
        }


class RAGSystem:
    """
    RAG (检索增强生成) 系统

    结合知识库和 LLM 提供架构建议
    """

    def __init__(
        self,
        knowledge_base: Optional[ArchitectureKnowledgeBase] = None,
        llm_agent: Optional[Any] = None,
    ):
        """
        Args:
            knowledge_base: 知识库
            llm_agent: LLM Agent
        """
        self.knowledge_base = knowledge_base or ArchitectureKnowledgeBase()
        self.llm_agent = llm_agent

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        检索相关知识

        Args:
            query: 查询
            top_k: 返回数量

        Returns:
            检索结果
        """
        return self.knowledge_base.search(query, top_k)

    def query_with_llm(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> Dict:
        """
        使用 LLM 生成回答（结合检索）

        Args:
            query: 用户查询
            context: 额外上下文

        Returns:
            LLM 回答
        """
        if self.llm_agent is None:
            return {
                'response': 'LLM agent not configured',
                'retrieved_context': self.retrieve(query),
            }

        # 检索相关知识
        retrieved = self.retrieve(query, top_k=3)
        context_str = '\n'.join([
            f"- {r['document']}"
            for r in retrieved
        ])

        # 构建 prompt
        prompt = f"""
Based on the following knowledge about industrial anomaly detection architectures:

{context_str}

{context or ''}

Question: {query}

Please provide a detailed answer based on the knowledge above.
"""

        # 调用 LLM
        response = self.llm_agent.generate(prompt)

        return {
            'response': response,
            'retrieved_context': retrieved,
        }

    def suggest_architecture(
        self,
        task_spec: Dict,
    ) -> Dict:
        """
        基于知识库建议架构

        Args:
            task_spec: 任务规格

        Returns:
            架构建议
        """
        # 检索相关知识
        query = f"anomaly detection {task_spec.get('category', 'industrial')}"
        relevant_knowledge = self.retrieve(query, top_k=5)

        # 提取关键信息
        successful_patterns = [
            r for r in relevant_knowledge
            if r['metadata'].get('category') == 'method'
        ]

        if successful_patterns:
            best = successful_patterns[0]
            return {
                'suggested_config': best['metadata'].get('performance', {}),
                'reasoning': best['document'],
                'confidence': best['score'],
                'knowledge_sources': len(relevant_knowledge),
            }

        return {
            'suggested_config': {},
            'reasoning': 'No directly applicable knowledge found, using default config',
            'confidence': 0.0,
            'knowledge_sources': 0,
        }


def create_rag_system(
    storage_path: str = './knowledge/vector_store.pkl',
    llm_agent: Optional[Any] = None,
) -> RAGSystem:
    """创建 RAG 系统的便捷函数"""
    kb = ArchitectureKnowledgeBase(storage_path=storage_path)
    return RAGSystem(knowledge_base=kb, llm_agent=llm_agent)
