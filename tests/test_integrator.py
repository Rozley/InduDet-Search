"""
搜索集成模块测试
"""

import tempfile

import pytest
import os

from src.search.integrator import SearchIntegrator, create_search_integrator
from src.search.predictor import PerformancePredictor
from src.llm.rag import ArchitectureKnowledgeBase


class TestSearchIntegrator:
    """测试搜索集成器"""

    def test_init_with_defaults(self):
        """测试默认初始化"""
        config = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_kb.pkl')
            integrator = SearchIntegrator(
                config=config,
                use_rag=True,
                use_predictor=True,
                use_augmentation=False,
                knowledge_path=path,
            )

            status = integrator.get_status()

            assert 'rag_enabled' in status
            assert 'predictor_enabled' in status

    def test_rag_suggestion(self):
        """测试 RAG 建议"""
        config = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_kb.pkl')
            integrator = SearchIntegrator(
                config=config,
                use_rag=True,
                use_predictor=False,
                knowledge_path=path,
            )

            suggestion = integrator.get_rag_suggestion({
                'category': 'bottle',
                'trial': 0,
            })

            assert 'suggested_config' in suggestion
            assert 'reasoning' in suggestion

    def test_predictor_performance(self):
        """测试性能预测"""
        config = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_kb.pkl')
            integrator = SearchIntegrator(
                config=config,
                use_rag=False,
                use_predictor=True,
                knowledge_path=path,
            )

            architecture = {
                'backbone': 'ResNet18',
                'method': 'memory_bank',
                'attention': 'CBAM',
            }

            prediction = integrator.get_predicted_performance(architecture)

            assert 'predicted_auroc' in prediction
            assert 'estimated_latency_ms' in prediction

    def test_early_stop(self):
        """测试早停判断"""
        config = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_kb.pkl')
            integrator = SearchIntegrator(
                config=config,
                use_rag=False,
                use_predictor=True,
                knowledge_path=path,
            )

            architecture = {'backbone': 'ResNet18'}

            # 测试低于阈值
            should_stop, reason = integrator.should_early_stop(
                architecture,
                current_fidelity='low',
                current_auroc=0.4,
            )

            assert should_stop is True

            # 测试高于阈值
            should_stop, reason = integrator.should_early_stop(
                architecture,
                current_fidelity='low',
                current_auroc=0.7,
            )

            assert should_stop is False

    def test_add_training_sample(self):
        """测试添加训练样本"""
        config = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_kb.pkl')
            integrator = SearchIntegrator(
                config=config,
                use_rag=False,
                use_predictor=True,
                knowledge_path=path,
            )

            architecture = {
                'backbone': 'ResNet18',
                'method': 'memory_bank',
            }

            metrics = {
                'auroc': 0.95,
                'latency_ms': 30,
                'params': 10e6,
            }

            integrator.add_training_sample(architecture, metrics)

            # 验证样本已添加
            assert len(integrator.predictor.training_data) == 1

    def test_knowledge_context(self):
        """测试获取知识上下文"""
        config = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_kb.pkl')
            integrator = SearchIntegrator(
                config=config,
                use_rag=True,
                use_predictor=False,
                knowledge_path=path,
            )

            context = integrator.get_knowledge_context("memory bank", top_k=2)

            # 应该有默认知识
            assert len(context) > 0


class TestCreateSearchIntegrator:
    """测试创建搜索集成器"""

    def test_create_function(self):
        """测试创建函数"""
        config = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_kb.pkl')
            integrator = create_search_integrator(
                config=config,
                knowledge_path=path,
                use_rag=True,
                use_predictor=True,
                use_augmentation=False,
            )

            assert integrator is not None
            status = integrator.get_status()
            assert 'rag_enabled' in status


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
