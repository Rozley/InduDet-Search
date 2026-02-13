"""
评估指标工具
计算异常检测相关的评估指标
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
)


def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    计算AUROC

    Args:
        scores: 预测分数
        labels: 真实标签 (0=正常, 1=异常)

    Returns:
        AUROC值
    """
    try:
        return roc_auc_score(labels, scores)
    except ValueError:
        # 只有一个类别
        return 0.5 if labels.mean() < 0.5 else 1.0


def compute_aupro(
    anomaly_maps: np.ndarray,
    gt_masks: np.ndarray,
    max_steps: int = 1000,
) -> float:
    """
    计算AUPRO (区域级AUROC)

    Args:
        anomaly_maps: 异常分数图 (N, H, W)
        gt_masks: 真实掩码 (N, H, W)
        max_steps: 最大采样步数

    Returns:
        AUPRO值
    """
    from scipy.ndimage import binary_dilation, binary_erosion

    pro_values = []
    thresholds = np.linspace(0, 1, max_steps)

    # 计算不同阈值下的recall
    for thresh in thresholds:
        predicted_masks = (anomaly_maps > thresh).astype(int)

        # 计算每个样本的recall
        recalls = []
        for pred_mask, gt_mask in zip(predicted_masks, gt_masks):
            if gt_mask.sum() == 0:
                # 没有异常区域
                recalls.append(1.0 if pred_mask.sum() == 0 else 0.0)
            else:
                # 计算区域重叠
                overlap = (pred_mask & gt_mask).sum()
                recall = overlap / gt_mask.sum()
                recalls.append(recall)

        pro_values.append(np.mean(recalls))

    # 计算AUPRO (PR曲线的积分)
    pro_values = np.array(pro_values)
    aupro = np.trapz(pro_values, thresholds) / thresholds.max()

    return float(aupro)


def compute_f1_max(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    计算最大F1分数

    Args:
        scores: 预测分数
        labels: 真实标签

    Returns:
        最大F1分数
    """
    thresholds = np.linspace(0, 1, 100)
    f1_scores = []

    for thresh in thresholds:
        preds = (scores >= thresh).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        f1_scores.append(f1)

    return float(max(f1_scores))


def compute_image_level_metrics(
    image_scores: np.ndarray,
    image_labels: np.ndarray,
) -> Dict[str, float]:
    """
    计算图像级评估指标

    Args:
        image_scores: 图像级异常分数 (N,)
        image_labels: 图像级标签 (N,)

    Returns:
        指标字典
    """
    auroc = compute_auroc(image_scores, image_labels)
    f1_max = compute_f1_max(image_scores, image_labels)

    try:
        ap = average_precision_score(image_labels, image_scores)
    except ValueError:
        ap = 0.5 if image_labels.mean() < 0.5 else 1.0

    return {
        'auroc': float(auroc),
        'ap': float(ap),
        'f1_max': float(f1_max),
    }


def compute_pixel_level_metrics(
    anomaly_maps: np.ndarray,
    gt_masks: np.ndarray,
) -> Dict[str, float]:
    """
    计算像素级评估指标

    Args:
        anomaly_maps: 异常分数图 (N, H, W)
        gt_masks: 真实掩码 (N, H, W)

    Returns:
        指标字典
    """
    # 展平
    scores_flat = anomaly_maps.flatten()
    labels_flat = gt_masks.flatten()

    # 移除全0或全1的位置
    valid_mask = labels_flat.sum() > 0
    if not valid_mask:
        return {
            'pixel_auroc': 0.5,
            'pixel_aupro': 0.5,
            'pixel_f1_max': 0.0,
        }

    auroc = compute_auroc(scores_flat, labels_flat)
    aupro = compute_aupro(anomaly_maps, gt_masks)
    f1_max = compute_f1_max(scores_flat, labels_flat)

    return {
        'pixel_auroc': float(auroc),
        'pixel_aupro': float(aupro),
        'pixel_f1_max': float(f1_max),
    }


def compute_efficiency_metrics(
    params: int,
    latency_ms: float,
    flops: Optional[int] = None,
    memory_mb: Optional[float] = None,
) -> Dict[str, float]:
    """
    计算效率指标

    Args:
        params: 参数量
        latency_ms: 推理延迟 (ms)
        flops: FLOPs
        memory_mb: 内存占用 (MB)

    Returns:
        指标字典
    """
    return {
        'params_m': params / 1e6,
        'latency_ms': latency_ms,
        'flops_g': flops / 1e9 if flops else None,
        'memory_mb': memory_mb,
    }


def compute_overall_score(
    metrics: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    计算综合分数

    Args:
        metrics: 指标字典
        weights: 各指标权重

    Returns:
        综合分数
    """
    default_weights = {
        'auroc': 0.4,
        'f1_max': 0.15,
        'params': 0.15,
        'latency': 0.20,
        'memory': 0.10,
    }

    if weights is None:
        weights = default_weights

    score = 0.0

    # 准确度指标
    if 'auroc' in metrics:
        # AUROC 0.5-1.0 映射到 0-1
        acc_score = max(0.0, (metrics['auroc'] - 0.5) * 2)
        score += weights.get('auroc', 0.4) * acc_score

    if 'f1_max' in metrics:
        score += weights.get('f1_max', 0.15) * metrics['f1_max']

    # 效率指标 (越小越好)
    if 'params' in metrics:
        params_score = 1.0 / (1.0 + np.log1p(metrics['params'] / 1e6))
        score += weights.get('params', 0.15) * params_score

    if 'latency' in metrics:
        latency_score = 1.0 / (1.0 + metrics['latency'] / 50)
        score += weights.get('latency', 0.20) * latency_score

    if 'memory' in metrics:
        memory_score = 1.0 / (1.0 + metrics['memory'] / 1024)
        score += weights.get('memory', 0.10) * memory_score

    return float(score)


def evaluate_anomaly_detector(
    model,
    test_loader,
    device: str = 'cuda',
) -> Dict[str, float]:
    """
    评估异常检测模型

    Args:
        model: 异常检测模型
        test_loader: 测试数据加载器
        device: 计算设备

    Returns:
        完整评估结果
    """
    model.eval()

    all_image_scores = []
    all_image_labels = []
    all_anomaly_maps = []
    all_gt_masks = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch[0].to(device)
            labels = batch[1].numpy()
            masks = batch[2]

            # 推理
            output = model.predict(images)

            # 收集图像级分数
            image_scores = output['image_score'].cpu().numpy()
            all_image_scores.extend(image_scores)
            all_image_labels.extend(labels)

            # 收集像素级分数
            anomaly_map = output['anomaly_map'].cpu().numpy()
            all_anomaly_maps.append(anomaly_map)

            if masks is not None:
                all_gt_masks.append(masks.numpy())

    # 计算图像级指标
    image_metrics = compute_image_level_metrics(
        np.array(all_image_scores),
        np.array(all_image_labels),
    )

    # 计算像素级指标
    pixel_metrics = {}
    if all_gt_masks:
        pixel_metrics = compute_pixel_level_metrics(
            np.concatenate(all_anomaly_maps),
            np.concatenate(all_gt_masks),
        )

    return {
        **image_metrics,
        **pixel_metrics,
    }
