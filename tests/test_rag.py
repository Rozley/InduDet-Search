"""
RAG 知识库系统测试
"""

import os
import tempfile

import pytest

from src.llm.rag import (
    SimpleVectorStore,
    ArchitectureKnowledgeBase,
    RAGSystem,
)


class TestSimpleVectorStore:
    """测试简单向量存储"""

    def test_add_and_search(self):
        """测试添加和搜索"""
        store = SimpleVectorStore()

        documents = [
            "PatchCore uses WideResNet50 with memory bank",
            "PaDiM uses ResNet50 with distribution based detection",
            "CBAM attention improves anomaly detection",
        ]

        store.add(documents)

        # 搜索
        results = store.search("memory bank", top_k=2)

        assert len(results) == 2
        assert results[0]['score'] > 0

    def test_empty_store(self):
        """测试空存储"""
        store = SimpleVectorStore()
        results = store.search("test", top_k=5)
        assert len(results) == 0

    def test_save_load(self):
        """测试保存和加载"""
        store = SimpleVectorStore()
        store.add(["test document"])

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name

        try:
            store.save(temp_path)

            new_store = SimpleVectorStore()
            new_store.load(temp_path)

            results = new_store.search("test", top_k=1)
            assert len(results) == 1
        finally:
            os.unlink(temp_path)


class TestArchitectureKnowledgeBase:
    """测试架构知识库"""

    def test_default_knowledge(self):
        """测试默认知识"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_kb.pkl')
            kb = ArchitectureKnowledgeBase(
                storage_path=path,
                use_default_knowledge=True,
            )

            stats = kb.get_statistics()
            assert stats['total_documents'] > 0

    def test_search(self):
        """测试搜索"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_kb.pkl')
            kb = ArchitectureKnowledgeBase(storage_path=path)

            # 添加测试知识
            kb.add_knowledge(
                content="WideResNet50 backbone achieves best performance",
                category='method',
                source='test',
                performance={'backbone': 'WideResNet50', 'auroc': 0.99},
            )

            # 搜索
            results = kb.search("WideResNet50 backbone", top_k=1)

            assert len(results) > 0
            assert 'WideResNet50' in results[0]['document']

    def test_add_architecture_experience(self):
        """测试添加架构经验"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_kb.pkl')
            kb = ArchitectureKnowledgeBase(storage_path=path)

            # 添加成功经验
            kb.add_architecture_experience(
                architecture={
                    'backbone': 'ResNet18',
                    'method': 'memory_bank',
                    'attention': 'CBAM',
                },
                metrics={'auroc': 0.95, 'latency_ms': 30, 'params': 10e6},
                is_success=True,
            )

            # 验证
            results = kb.get_successful_patterns()
            assert len(results) >= 1

    def test_search_with_filter(self):
        """测试带过滤的搜索"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_kb.pkl')
            kb = ArchitectureKnowledgeBase(storage_path=path)

            kb.add_knowledge("test insight", category='insight', source='test')
            kb.add_knowledge("test method", category='method', source='test')

            results = kb.search("test", category_filter='insight', top_k=5)

            for r in results:
                assert r['metadata']['category'] == 'insight'


class TestRAGSystem:
    """测试 RAG 系统"""

    def test_retrieve(self):
        """测试检索"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_rag.pkl')
            rag = RAGSystem(
                knowledge_base=ArchitectureKnowledgeBase(storage_path=path)
            )

            results = rag.retrieve("memory bank", top_k=3)
            assert len(results) <= 3

    def test_suggest_architecture(self):
        """测试架构建议"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_rag.pkl')
            kb = ArchitectureKnowledgeBase(storage_path=path)

            # 添加一些知识
            kb.add_knowledge(
                content="PatchCore uses WideResNet50 with memory bank achieving 99.6% AUROC",
                category='method',
                source='test',
                performance={'auroc': 0.996, 'method': 'memory_bank'},
            )

            rag = RAGSystem(knowledge_base=kb)

            suggestion = rag.suggest_architecture({'category': 'bottle'})

            assert 'suggested_config' in suggestion
            assert 'reasoning' in suggestion


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
