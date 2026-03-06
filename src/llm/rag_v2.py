"""
RAG 知识库系统 V2
基于 Chroma + Embeddings 的增强版 RAG
参考 SR-LLM 设计，增加了 reflection 机制
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


class Knowledge:
    """
    知识单元

    参考 SR-LLM 的知识结构:
    - source: 知识来源 (drll, experience, paper)
    - target: 目标/应用场景
    - key: 关键词/检索键
    - content: 知识内容
    - comment: 人工评价
    - reflection: AI 反思总结
    """

    def __init__(
        self,
        source: str,
        target: str,
        key: str,
        content: str,
        comment: Optional[str] = None,
        reflection: Optional[str] = None,
        performance: Optional[Dict] = None,
    ):
        """
        Args:
            source: 知识来源 (drll, experience, paper, success, failure)
            target: 目标符号/架构
            key: 检索关键词
            content: 知识内容
            comment: 人工评价/注释
            reflection: AI 反思总结
            performance: 性能指标
        """
        self.id = str(uuid.uuid4())
        self.source = source
        self.target = target
        self.key = key
        self.content = content
        self.comment = comment or ""
        self.reflection = reflection or ""
        self.performance = performance or {}
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'id': self.id,
            'source': self.source,
            'target': self.target,
            'key': self.key,
            'content': self.content,
            'comment': self.comment,
            'reflection': self.reflection,
            'performance': self.performance,
            'timestamp': self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Knowledge':
        """从字典创建"""
        knowledge = cls(
            source=data.get('source', 'unknown'),
            target=data.get('target', ''),
            key=data.get('key', ''),
            content=data.get('content', ''),
            comment=data.get('comment'),
            reflection=data.get('reflection'),
            performance=data.get('performance'),
        )
        if 'id' in data:
            knowledge.id = data['id']
        if 'timestamp' in data:
            knowledge.timestamp = data['timestamp']
        return knowledge


class ChromaKnowledgePool:
    """
    Chroma 向量知识库

    使用 Chroma + Embeddings 实现语义检索
    """

    def __init__(
        self,
        storage_path: str = './knowledge/chroma_db',
        embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
    ):
        """
        Args:
            storage_path: Chroma 数据库存储路径
            embedding_model: embedding 模型名称
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.embedding_model = embedding_model
        self._init_embeddings()
        self._init_chroma()

        # 知识库统计
        self._load_metadata()

    def _init_embeddings(self):
        """初始化 embeddings"""
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cpu'},
            )
            self.use_langchain = True
        except ImportError:
            # 回退到简单的 hash-based 方法
            print("Warning: langchain not installed, using simple embeddings")
            self.use_langchain = False
            self.embeddings = None

    def _init_chroma(self):
        """初始化 Chroma"""
        try:
            import chroma
            from langchain_community.vectorstores import Chroma
            self.db = Chroma(
                persist_directory=str(self.storage_path),
                embedding_function=self.embeddings if self.use_langchain else None,
            )
            self.use_chroma = True
        except ImportError:
            print("Warning: Chroma not installed, using simple in-memory store")
            self.use_chroma = False
            self.db = self._create_simple_store()

    def _create_simple_store(self):
        """创建简单的内存存储作为回退"""
        return {
            'documents': [],
            'metadatas': [],
            'ids': [],
        }

    def _load_metadata(self):
        """加载元数据"""
        metadata_path = self.storage_path / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'total_count': 0,
                'sources': {},
                'last_update': None,
            }

    def _save_metadata(self):
        """保存元数据"""
        metadata_path = self.storage_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def add_knowledge(
        self,
        knowledge: Knowledge,
    ) -> str:
        """
        添加知识

        Args:
            knowledge: 知识单元

        Returns:
            知识 ID
        """
        doc_id = knowledge.id

        if self.use_chroma and self.use_langchain:
            from langchain.schema import Document
            doc = Document(
                page_content=knowledge.key,
                metadata=knowledge.to_dict(),
            )
            self.db.add_documents([doc], [doc_id])
        else:
            # 简单存储
            self.db['documents'].append(knowledge.key)
            self.db['metadatas'].append(knowledge.to_dict())
            self.db['ids'].append(doc_id)

        # 更新元数据
        self.metadata['total_count'] += 1
        source = knowledge.source
        self.metadata['sources'][source] = self.metadata['sources'].get(source, 0) + 1
        self.metadata['last_update'] = datetime.now().isoformat()
        self._save_metadata()

        return doc_id

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        source_filter: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[Dict]:
        """
        检索知识

        Args:
            query: 查询文本
            top_k: 返回数量
            source_filter: 来源过滤
            min_score: 最小相似度

        Returns:
            检索结果列表
        """
        if self.use_chroma and self.use_langchain:
            results = self.db.similarity_search_with_score(query, k=top_k)
            retrieved = []
            for doc, score in results:
                if score >= min_score:
                    metadata = doc.metadata
                    if source_filter is None or metadata.get('source') == source_filter:
                        retrieved.append({
                            'knowledge': Knowledge.from_dict(metadata),
                            'score': float(score),
                            'document': doc.page_content,
                        })
            return retrieved
        else:
            # 简单回退：返回所有知识
            retrieved = []
            for i, doc in enumerate(self.db['documents']):
                # 简单的关键词匹配
                score = self._simple_similarity(query, doc)
                if score >= min_score:
                    metadata = self.db['metadatas'][i]
                    if source_filter is None or metadata.get('source') == source_filter:
                        retrieved.append({
                            'knowledge': Knowledge.from_dict(metadata),
                            'score': score,
                            'document': doc,
                        })
            # 按分数排序
            retrieved.sort(key=lambda x: x['score'], reverse=True)
            return retrieved[:top_k]

    def _simple_similarity(self, query: str, document: str) -> float:
        """简单的相似度计算"""
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        if not query_words or not doc_words:
            return 0.0
        intersection = query_words.intersection(doc_words)
        return len(intersection) / len(query_words)

    def get_by_source(self, source: str) -> List[Knowledge]:
        """按来源获取知识"""
        results = []
        if self.use_chroma and self.use_langchain:
            # 使用过滤查询
            docs = self.db.get(where={'source': source})
            for i, metadata in enumerate(docs.get('metadatas', [])):
                results.append(Knowledge.from_dict(metadata))
        else:
            for metadata in self.db['metadatas']:
                if metadata.get('source') == source:
                    results.append(Knowledge.from_dict(metadata))
        return results

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'total_count': self.metadata['total_count'],
            'sources': self.metadata['sources'],
            'last_update': self.metadata['last_update'],
        }

    def delete(self, knowledge_id: str):
        """删除知识"""
        if self.use_chroma and self.use_langchain:
            self.db.delete(ids=[knowledge_id])
        else:
            for i, id_ in enumerate(self.db['ids']):
                if id_ == knowledge_id:
                    self.db['documents'].pop(i)
                    self.db['metadatas'].pop(i)
                    self.db['ids'].pop(i)
ids'].pop(i)
                    break

    def clear(self):
        """清空知识库"""
        if self.use_chroma and self.use_langchain:
            self.db.delete(where={})
        else:
            self.db['documents'].clear()
            self.db['metadatas'].clear()
            self.db['ids'].clear()
        self.metadata = {'total_count': 0, 'sources': {}, 'last_update': None}
        self._save_metadata()


class EnhancedRAGSystem:
    """
    增强版 RAG 系统

    功能:
    - Chroma 向量检索
    - Reflection 机制
    - 知识库自动更新
    """

    def __init__(
        self,
        storage_path: str = './knowledge/chroma_db',
        llm_agent: Optional[Any] = None,
        embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
    ):
        """
        Args:
            storage_path: 知识库存储路径
            llm_agent: LLM Agent (用于 reflection)
            embedding_model: Embedding 模型
        """
        self.knowledge_pool = ChromaKnowledgePool(
            storage_path=storage_path,
            embedding_model=embedding_model,
        )
        self.llm_agent = llm_agent

    def add_experience(
        self,
        architecture: Dict,
        metrics: Dict,
        is_success: bool = True,
        comment: Optional[str] = None,
    ) -> Knowledge:
        """
        添加架构搜索经验

        Args:
            architecture: 架构配置
            metrics: 性能指标
            is_success: 是否成功
            comment: 评价

        Returns:
            添加的知识
        """
        # 构建内容
        backbone = architecture.get('backbone', 'unknown')
        method = architecture.get('method', 'unknown')
        attention = architecture.get('attention', 'none')

        if is_success:
            content = (
                f"Successful architecture: {backbone} backbone with {method} method "
                f"and {attention} attention. Achieved AUROC: {metrics.get('auroc', 0):.4f}, "
                f"latency: {metrics.get('latency_ms', 0):.2f}ms"
            )
            source = 'success'
            target = f"{backbone}_{method}"
        else:
            content = (
                f"Failed architecture: {backbone} backbone with {method} method. "
                f"AUROC: {metrics.get('auroc', 0):.4f}. Should avoid this configuration."
            )
            source = 'failure'
            target = f"{backbone}_{method}"

        # 构建 key
        key = f"{backbone} {method} {attention}"

        knowledge = Knowledge(
            source=source,
            target=target,
            key=key,
            content=content,
            comment=comment or ("Good configuration" if is_success else "Poor configuration"),
            performance={
                'auroc': metrics.get('auroc', 0),
                'latency_ms': metrics.get('latency_ms', 0),
                'params': metrics.get('params', 0),
                'is_success': is_success,
            },
        )

        self.knowledge_pool.add_knowledge(knowledge)
        return knowledge

    def add_knowledge(self, knowledge: Knowledge):
        """直接添加知识"""
        self.knowledge_pool.add_knowledge(knowledge)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        source_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        检索知识

        Args:
            query: 查询文本
            top_k: 返回数量
            source_filter: 来源过滤

        Returns:
            检索结果
        """
        return self.knowledge_pool.retrieve(
            query=query,
            top_k=top_k,
            source_filter=source_filter,
        )

    async def reflection(
        self,
        architecture: Dict,
        metrics: Dict,
    ) -> Optional[Knowledge]:
        """
        Reflection 机制：让 LLM 反思成功/失败的架构

        Args:
            architecture: 架构配置
            metrics: 性能指标

        Returns:
            反思后的知识
        """
        if self.llm_agent is None:
            return None

        is_success = metrics.get('auroc', 0) >= 0.85

        # 构建反思 prompt
        prompt = f"""
作为工业异常检测架构设计专家，请分析以下架构的性能并提供设计洞察。

架构配置:
- Backbone: {architecture.get('backbone', 'unknown')}
- Method: {architecture.get('method', 'unknown')}
- Attention: {architecture.get('attention', 'none')}
- Feature Levels: {architecture.get('levels', 'unknown')}
- Memory Size: {architecture.get('memory_size', 'unknown')}

性能指标:
- AUROC: {metrics.get('auroc', 0):.4f}
- Latency: {metrics.get('latency_ms', 0):.2f}ms
- Parameters: {metrics.get('params', 0):,}

请提供:
1. 为什么这个配置成功/失败
2. 关键设计要点
3. 对未来搜索的建议

请用中文回复。
"""

        try:
            response = self.llm_agent.generate(prompt)

            # 创建反思知识
            knowledge = Knowledge(
                source='reflection',
                target=architecture.get('backbone', 'unknown'),
                key=f"{architecture.get('backbone')}_{architecture.get('method')}",
                content=str(response),
                reflection=prompt,
                performance={
                    'auroc': metrics.get('auroc', 0),
                    'is_success': is_success,
                },
            )

            self.knowledge_pool.add_knowledge(knowledge)
            return knowledge

        except Exception as e:
            print(f"Reflection failed: {e}")
            return None

    def get_successful_patterns(self, top_k: int = 10) -> List[Dict]:
        """获取成功模式"""
        return self.knowledge_pool.retrieve(
            query="successful architecture",
            top_k=top_k,
        )

    def get_failed_patterns(self, top_k: int = 10) -> List[Dict]:
        """获取失败模式"""
        return self.knowledge_pool.retrieve(
            query="failed architecture",
            top_k=top_k,
        )

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = self.knowledge_pool.get_statistics()
        stats['success_count'] = len(self.get_successful_patterns(top_k=100))
        stats['failure_count'] = len(self.get_failed_patterns(top_k=100))
        return stats


def create_enhanced_rag(
    storage_path: str = './knowledge/chroma_db',
    llm_agent: Optional[Any] = None,
) -> EnhancedRAGSystem:
    """创建增强 RAG 系统的便捷函数"""
    return EnhancedRAGSystem(
        storage_path=storage_path,
        llm_agent=llm_agent,
    )
