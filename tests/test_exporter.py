"""
模型导出模块测试
"""

import os
import tempfile

import pytest
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """简单的测试模型"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


from src.utils.exporter import ModelExporter, export_model, get_model_size


class TestModelExporter:
    """测试模型导出器"""

    def test_init(self):
        """测试初始化"""
        model = SimpleModel()
        exporter = ModelExporter(
            model=model,
            input_shape=(1, 3, 224, 224),
            output_path='./test_exports',
        )

        assert exporter.model is not None
        assert exporter.input_shape == (1, 3, 224, 224)

    def test_export_torchscript(self):
        """测试 TorchScript 导出"""
        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ModelExporter(
                model=model,
                input_shape=(1, 3, 64, 64),
                output_path=tmpdir,
            )

            output_path = exporter.export_torchscript('test_model.pt')

            assert os.path.exists(output_path)
            assert output_path.endswith('.pt')

    def test_export_onnx(self):
        """测试 ONNX 导出"""
        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ModelExporter(
                model=model,
                input_shape=(1, 3, 64, 64),
                output_path=tmpdir,
            )

            output_path = exporter.export_onnx('test_model.onnx', simplify=False)

            assert os.path.exists(output_path)
            assert output_path.endswith('.onnx')

    def test_export_onnx_inference(self):
        """测试 ONNX 导出后可正常推理"""
        model = SimpleModel()
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ModelExporter(
                model=model,
                input_shape=(1, 3, 64, 64),
                output_path=tmpdir,
            )

            output_path = exporter.export_onnx('test_model.onnx', simplify=False)

            # 验证 ONNX 模型
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)

            # 测试推理
            import onnxruntime as ort
            sess = ort.InferenceSession(output_path)
            input_name = sess.get_inputs()[0].name

            dummy_input = np.random.randn(1, 3, 64, 64).astype(np.float32)
            output = sess.run(None, {input_name: dummy_input})

            assert output[0].shape == (1, 10)


class TestExportModel:
    """测试便捷导出函数"""

    def test_export_onnx_function(self):
        """测试 ONNX 导出函数"""
        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = export_model(
                model=model,
                output_path=tmpdir,
                format='onnx',
                input_shape=(1, 3, 64, 64),
            )

            assert os.path.exists(output_path)

    def test_export_torchscript_function(self):
        """测试 TorchScript 导出函数"""
        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = export_model(
                model=model,
                output_path=tmpdir,
                format='torchscript',
                input_shape=(1, 3, 64, 64),
            )

            assert os.path.exists(output_path)


class TestGetModelSize:
    """测试获取模型大小"""

    def test_existing_file(self):
        """测试获取已存在文件的大小"""
        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = export_model(
                model=model,
                output_path=tmpdir,
                format='torchscript',
                input_shape=(1, 3, 64, 64),
            )

            size_info = get_model_size(output_path)

            assert 'size_bytes' in size_info
            assert 'size_mb' in size_info
            assert size_info['size_bytes'] > 0

    def test_nonexistent_file(self):
        """测试获取不存在文件的大小"""
        size_info = get_model_size('/nonexistent/model.pt')

        assert 'error' in size_info


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
