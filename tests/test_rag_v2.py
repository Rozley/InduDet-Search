"""
增强版 RAG 系统测试
"""

import os
import tempfile

import pytest
import numpy as np

from src.llm.rag_v2 import (
    Knowledge,
    ChromaKnowledgePool,
    EnhancedRAGSystem,
    create_enhanced_rag,
)


class TestKnowledge:
    """测试知识单元"""

    def test_create_knowledge(self):
        """测试创建知识"""
        knowledge = Knowledge(
            source='success',
            target='ResNet18_memory_bank',
            key='ResNet18 memory_bank CBAM',
            content='Successful architecture',
            comment='Good config',
            reflection='Insight',
            performance={'auroc': 0.95},
        )

        assert knowledge.source == 'success'
        assert knowledge.target == 'ResNet18_memory_bank'
        assert 'id' in knowledge.to_dict()
        assert 'timestamp' in knowledge.to_dict()

    def test_to_dict_from_dict(self):
        """测试序列化"""
        original = Knowledge(
            source='failure',
            target='test',
            key='key',
            content='content',
            performance={'auroc': 0.5},
        )

        data = original.to_dict()
        restored = Knowledge.from_dict(data)

        assert restored.source == original.source
        assert restored.target == original.target
        assert restored.performance == original.performance


class TestChromaKnowledgePool:
    """测试 Chroma 知识库"""

    def test_add_and_retrieve(self):
        """测试添加和检索"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = ChromaKnowledgePool(
                storage_path=tmpdir,
                embedding_model='sentence-transformers/all-MiniLM-L6-v2',
            )

            # 添加知识
            knowledge = Knowledge(
                source='success',
                target='ResNet18',
                key='ResNet18 memory_bank',
                content='Successful with AUROC 0.95',
            )
            pool.add_knowledge(knowledge)

            # 检索
            results = pool.retrieve('memory bank', top_k=1)

            # 应该有结果（如果使用简单模式）
            assert pool.get_statistics()['total_count'] >= 1

    def test_statistics(self):
        """测试统计信息"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = ChromaKnowledgePool(storage_path=tmpdir)

            # 添加多个知识
            for i in range(3):
                k = Knowledge(
                    source='success' if i < 2 else 'failure',
                    target=f'test_{i}',
                    key=f'key_{i}',
                    content=f'content_{i}',
                )
                pool.add_knowledge(k)

            stats = pool.get_statistics()
            assert stats['total_count'] == 3
            assert stats['sources']['success'] == 2
            assert stats['sources']['failure'] == 1


class TestEnhancedRAGSystem:
    """测试增强 RAG 系统"""

    def test_add_experience(self):
        """测试添加经验"""
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = EnhancedRAGSystem(storage_path=tmpdir)

            # 添加成功经验
            knowledge = rag.add_experience(
                architecture={
                    'backbone': 'ResNet18',
                    'method': 'memory_bank',
                    'attention': 'CBAM',
                },
                metrics={'auroc': 0.95, 'latency_ms': 30, 'params': 10e6},
                is_success=True,
            )

            assert knowledge.source == 'success'
            assert 'ResNet18' in knowledge.key

    def test_add_failure_experience(self):
        """测试添加失败经验"""
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = EnhancedRAGSystem(storage_path=tmpdir)

            knowledge = rag.add_experience(
                architecture={
                    'backbone': 'ResNet18',
                    'method': 'distribution',
                },
                metrics={'auroc': 0.4, 'latency_ms': 100, 'params': 20e6},
                is_success=False,
            )

            assert knowledge.source == 'failure'
            assert knowledge.performance['is_success'] is False

    def test_retrieve(self):
        """测试检索"""
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = EnhancedRAGSystem(storage_path=tmpdir)

            # 添加一些知识
            rag.add_experience(
                {'backbone': 'ResNet18', 'method': 'memory_bank'},
                {'auroc': 0.95},
                is_success=True,
            )

            # 检索
            results = rag.retrieve('ResNet18 memory_bank', top_k=3)

            # 应该有结果
            assert isinstance(results, list)

    def test_get_successful_patterns(self):
        """测试获取成功模式"""
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = EnhancedRAGSystem(storage_path=tmpdir)

            # 添加成功和失败经验
            rag.add_experience(
                {'backbone': 'ResNet18', 'method': 'memory_bank'},
                {'auroc': 0.95},
                is_success=True,
            )
            rag.add_experience(
                {'backbone': 'MobileNetV3', 'method': 'distribution'},
                {'auroc': 0.4},
                is_success=False,
            )

            success = rag.get_successful_patterns(top_k=5)
            assert isinstance(success, list)


class TestCreateEnhancedRAG:
    """测试创建函数"""

    def test_create_function(self):
        """测试创建函数"""
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = create_enhanced_rag(storage_path=tmpdir)
            assert rag is not None
            assert rag.knowledge_pool is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
