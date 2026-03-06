"""
模型导出模块
支持 TensorRT、ONNX、TFLite 等格式导出
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union


class ModelExporter:
    """
    模型导出器

    支持多种格式导出:
    - ONNX: 通用格式
    - TensorRT: NVIDIA GPU 优化
    - TFLite: 边缘设备
    """

    def __init__(
        self,
        model: 'torch.nn.Module',
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        output_path: str = './exports',
    ):
        """
        Args:
            model: PyTorch 模型
            input_shape: 输入形状 (C, H, W)
            output_path: 导出目录
        """
        self.model = model
        self.input_shape = input_shape
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # 设置为评估模式
        self.model.eval()

    def export_onnx(
        self,
        filename: str = 'model.onnx',
        opset_version: int = 11,
        simplify: bool = True,
    ) -> str:
        """
        导出为 ONNX 格式

        Args:
            filename: 文件名
            opset_version: ONNX 操作集版本
            simplify: 是否简化

        Returns:
            导出文件路径
        """
        import torch

        output_path = self.output_path / filename

        # 创建示例输入
        dummy_input = torch.randn(*self.input_shape)

        # 导出
        torch.onnx.export(
            self.model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        # 简化 (可选)
        if simplify:
            try:
                import onnx
                from onnxsim import simplify

                onnx_model = onnx.load(str(output_path))
                model_simp, check = simplify(onnx_model)
                onnx.save(model_simp, str(output_path))
                print(f"ONNX model simplified")
            except ImportError:
                print("onnxsim not installed, skipping simplification")

        print(f"ONNX model exported to: {output_path}")
        return str(output_path)

    def export_torchscript(
        self,
        filename: str = 'model.pt',
    ) -> str:
        """
        导出为 TorchScript 格式

        Args:
            filename: 文件名

        Returns:
            导出文件路径
        """
        import torch

        output_path = self.output_path / filename

        # 追踪
        dummy_input = torch.randn(*self.input_shape)
        traced_model = torch.jit.trace(self.model, dummy_input)
        traced_model.save(str(output_path))

        print(f"TorchScript model exported to: {output_path}")
        return str(output_path)

    def export_tflite(
        self,
        filename: str = 'model.tflite',
        quantize: bool = True,
    ) -> str:
        """
        导出为 TFLite 格式

        Args:
            filename: 文件名
            quantize: 是否量化

        Returns:
            导出文件路径
        """
        try:
            import torch
            import onnx
            import tensorflow as tf
        except ImportError as e:
            print(f"Required package not installed: {e}")
            print("Install with: pip install tensorflow onnx-tf")
            return None

        output_path = self.output_path / filename

        # 先导出 ONNX
        onnx_path = self.output_path / 'temp_model.onnx'
        self.export_onnx('temp_model.onnx', simplify=False)

        # 转换为 TFLite
        onnx_model = onnx.load(str(onnx_path))
        tf_rep = tf.compat.v1.lite.TFLiteConverter.from_onnx(onnx_model)

        if quantize:
            tf_rep.optimizations = [tf.lite.Optimize.DEFAULT]
            tf_rep.target_spec.supported_types = [tf.float16]

        tflite_model = tf_rep.convert()

        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        # 清理临时文件
        onnx_path.unlink()

        print(f"TFLite model exported to: {output_path}")
        return str(output_path)

    def export_tensorrt(
        self,
        filename: str = 'model.trt',
        precision: str = 'fp32',
        workspace_size: int = 1 << 30,
    ) -> str:
        """
        导出为 TensorRT 格式

        Args:
            filename: 文件名
            precision: 精度 ('fp32', 'fp16', 'int8')
            workspace_size: 工作空间大小 (字节)

        Returns:
            导出文件路径
        """
        try:
            import tensorrt as trt
            import torch
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError as e:
            print(f"Required package not installed: {e}")
            print("Install with: pip install tensorrt")
            return None

        output_path = self.output_path / filename

        # 创建 TensorRT logger
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)

        # 创建网络
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        config = builder.create_builder_config()
        config.max_workspace_size = workspace_size

        # 解析 ONNX
        onnx_path = self.output_path / 'temp_model.onnx'
        self.export_onnx('temp_model.onnx', simplify=False)

        parser = trt.OnnxParser(network, logger)
        with open(onnx_path, 'rb') as f:
            parser.parse(f.read())

        # 构建引擎
        if precision == 'fp16':
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == 'int8':
            config.set_flag(trt.BuilderFlag.INT8)

        engine = builder.build_serialized_network(network, config)

        # 保存
        with open(output_path, 'wb') as f:
            f.write(engine)

        # 清理
        onnx_path.unlink()

        print(f"TensorRT model exported to: {output_path}")
        return str(output_path)

    def export_pretrained(
        self,
        format: str = 'onnx',
        **kwargs,
    ) -> str:
        """
        便捷导出方法

        Args:
            format: 格式 ('onnx', 'torchscript', 'tflite', 'tensorrt')
            **kwargs: 其他参数

        Returns:
            导出文件路径
        """
        if format == 'onnx':
            return self.export_onnx(**kwargs)
        elif format == 'torchscript':
            return self.export_torchscript(**kwargs)
        elif format == 'tflite':
            return self.export_tflite(**kwargs)
        elif format == 'tensorrt':
            return self.export_tensorrt(**kwargs)
        else:
            raise ValueError(f"Unknown format: {format}")


def export_model(
    model: 'torch.nn.Module',
    output_path: str,
    format: str = 'onnx',
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    **kwargs,
) -> str:
    """
    导出模型的便捷函数

    Args:
        model: PyTorch 模型
        output_path: 输出目录
        format: 导出格式
        input_shape: 输入形状
        **kwargs: 其他参数

    Returns:
        导出文件路径
    """
    exporter = ModelExporter(
        model=model,
        input_shape=input_shape,
        output_path=output_path,
    )

    return exporter.export_pretrained(format=format, **kwargs)


def get_model_size(model_path: str) -> Dict:
    """
    获取模型文件大小信息

    Args:
        model_path: 模型文件路径

    Returns:
        大小信息字典
    """
    path = Path(model_path)
    if not path.exists():
        return {'error': 'File not found'}

    size_bytes = path.stat().st_size

    return {
        'path': str(path),
        'size_bytes': size_bytes,
        'size_mb': size_bytes / (1024 * 1024),
        'size_mb_rounded': round(size_bytes / (1024 * 1024), 2),
    }
