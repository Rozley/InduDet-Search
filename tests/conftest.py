"""测试配置文件"""

import pytest


def pytest_configure(config):
    """pytest 配置"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


def pytest_collection_modifyitems(config, items):
    """修改测试收集"""
    for item in items:
        # 为所有测试添加默认标记
        item.add_marker(pytest.mark.unit)
