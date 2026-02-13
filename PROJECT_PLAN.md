# LLM+RAG+RL 工业异常检测架构搜索系统

## 一、项目概述

### 背景与目标
设计一个结合LLM（大型语言模型）、RAG（检索增强生成）、RL（强化学习）和NAS（神经架构搜索）的自动化系统，用于搜索高效的工业异常检测模型架构。

### 核心创新点
- **LLM作为架构设计Agent**：利用大语言模型的知识进行架构设计决策
- **RAG检索增强**：检索学术论文、架构案例、历史搜索经验
- **RL驱动NAS搜索**：使用强化学习控制器自动生成和优化架构
- **工业异常检测专用**：针对MVTec AD等数据集优化

---

## 二、技术架构

### 2.1 系统组件图

```
┌─────────────────────────────────────────────────────────────────────┐
│                     InduDet-Search 系统架构                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │
│  │   LLM Agent  │◄──►│     RAG      │◄──►│  知识库      │         │
│  │  (架构设计)   │    │  (检索增强)   │    │  (论文/案例)  │         │
│  └──────┬───────┘    └──────┬───────┘    └──────────────┘         │
│         │                   │                                      │
│         ▼                   ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  RL Controller (强化学习控制器)               │   │
│  │  ┌─────────────────────────────────────────────────────┐   │   │
│  │  │  RNN/LSTM Controller                               │   │   │
│  │  │  ├── 生成架构编码 (architecture encoding)           │   │   │
│  │  │  ├── 策略梯度优化 (policy gradient)                │   │   │
│  │  │  └── 奖励信号处理 (reward shaping)                 │   │   │
│  │  └─────────────────────────────────────────────────────┘   │   │
│  └───────────────────────────┬─────────────────────────────────┘   │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Architecture Evaluator                    │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │   │
│  │  │ SuperNet    │  │  Weight     │  │ Performance         │ │   │
│  │  │ (超网络)     │  │  Sharing    │  │ Predictor           │ │   │
│  │  └─────────────┘  │  (权重共享)  │  │ (性能预测)           │ │   │
│  │                   └─────────────┘  └─────────────────────┘ │   │
│  └───────────────────────────┬─────────────────────────────────┘   │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │               Industrial Anomaly Search Space               │   │
│  │  ├── Encoder: ResNet18/50, EfficientNet, ViT              │   │
│  │  ├── Attention: SE, CBAM, TripletAttention                │   │
│  │  ├── MemoryBank: Coreset Sampling, k-NN                   │   │
│  │  ├── Scoring: Mahalanobis, Cosine, Euclidean              │   │
│  │  └── Loss: Reconstruction, Contrastive, Focal             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                       评估 & 部署                           │   │
│  │  ├── Multi-Fidelity Evaluation (低保真→高保真)              │   │
│  │  ├── Early Stopping (早停机制)                              │   │
│  │  └── TFLite/ONNX Export (模型导出)                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 三、搜索空间设计

### 3.1 编码器搜索空间 (Encoder Search Space)

```python
ENCODER_OPTIONS = {
    'backbone': [
        'ResNet18', 'ResNet34', 'ResNet50',
        'WideResNet50',
        'EfficientNet-B0', 'EfficientNet-B3',
        'MobileNetV3',
        'ViT-Small', 'ViT-Base',
        'Swin-T', 'Swin-S'
    ],
    'pretrained': [True, False],
    'freeze_backbone': [True, False],
    'drop_rate': [0.0, 0.1, 0.2, 0.3]
}
```

### 3.2 特征提取搜索空间 (Feature Extraction)

```python
FEATURE_OPTIONS = {
    'levels': [
        [2], [2, 3], [1, 2, 3], [1, 2, 3, 4],  # ResNet stages
    ],
    'attention_modules': [
        'none', 'SE', 'CBAM', 'ECA', 'TripletAttention'
    ],
    'feature_dim': [64, 128, 256, 512],
    'pooling': ['avg', 'max', 'adaptive', 'identity']
}
```

### 3.3 检测头搜索空间 (Detection Head)

```python
HEAD_OPTIONS = {
    'method': [
        'memory_bank',      # PatchCore style
        'distribution',     # PaDiM style
        'student_teacher',  # Knowledge distillation
        'contrastive'       # CSI style
    ],
    'memory_bank': {
        'size': [100, 500, 1000, 5000],
        'sampling': ['random', 'kcenter', 'kmeans', 'herding'],
        'reduction': ['none', 'pca', 'random_projection']
    },
    'scoring': {
        'distance': ['euclidean', 'cosine', 'mahalanobis'],
        'k': [1, 3, 5, 9, 21],
        'temperature': [0.1, 0.5, 1.0, 2.0]
    }
}
```

### 3.4 训练策略搜索空间

```python
TRAINING_OPTIONS = {
    'epochs': [50, 100, 200, 300],
    'batch_size': [8, 16, 32, 64],
    'optimizer': ['adam', 'sgd', 'adamw'],
    'learning_rate': [1e-5, 1e-4, 1e-3],
    'weight_decay': [0.0, 1e-4, 1e-3],
    'scheduler': ['cosine', 'step', 'plateau', 'none']
}
```

### 3.5 数据增强搜索空间 (异常生成算法) ⭐

```python
AUGMENTATION_OPTIONS = {
    # 合成异常生成方法
    'anomaly_generation': [
        'cutpaste',           # CutPaste: 裁剪-粘贴异常
        'copy_paste',         # Copy-Paste: 复制粘贴
        'mixup',              # MixUp: 混合增强
        'cutmix',             # CutMix: 裁剪混合
        'paste_gan',          # GAN生成异常粘贴
        'diffusion',          # Diffusion模型生成异常
        'semantic_shift',     # 语义分割+替换
        'style_transfer',     # 风格迁移异常
        'noise_injection',   # 噪声注入
        'blur_sharpness',    # 模糊/清晰度异常
    ],

    # 异常强度参数
    'anomaly_intensity': [0.1, 0.2, 0.3, 0.5],  # 异常区域占比
    'anomaly_count': [1, 3, 5],                  # 每张图的异常数量

    # 异常类型
    'anomaly_types': [
        'point',              # 点异常
        'line',               # 线条异常
        'patch',             # 块状异常
        'region',            # 区域异常
        'texture',           # 纹理异常
        'color',             # 颜色异常
    ],

    # 正常样本增强
    'normal_augmentation': [
        'none',              # 不增强
        'standard',          # 标准增强(翻转、旋转、颜色抖动)
        'strong',            # 强增强(RandAugment)
    ]
}


class AnomalyAugmentor:
    """
    异常生成器 - 用于数据增强

    支持多种异常生成算法:
    1. CutPaste: 裁剪正常区域并粘贴到其他位置
    2. Diffusion-based: 使用扩散模型生成更真实的异常
    3. GAN-based: 使用GAN生成逼真异常
    4. Copy-Paste: 复制粘贴增强
    """

    def __init__(self, method: str = 'cutpaste', intensity: float = 0.3):
        self.method = method
        self.intensity = intensity
        self._init_augmentor()

    def _init_augmentor(self):
        """初始化增强器"""
        if self.method == 'cutpaste':
            self._init_cutpaste()
        elif self.method == 'copy_paste':
            self._init_copy_paste()
        elif self.method == 'diffusion':
            self._init_diffusion()
        elif self.method == 'paste_gan':
            self._init_gan()

    def _init_cutpaste(self):
        """CutPaste初始化"""
        self.crop_sizes = [0.1, 0.2, 0.3]
        self.crop_scales = [0.5, 1.0, 1.5]

    def _init_copy_paste(self):
        """Copy-Paste初始化"""
        self.blend_modes = ['alpha', 'poisson', 'none']
        self.jitter = 5

    def generate_anomaly(self, image: np.ndarray) -> np.ndarray:
        """在正常图像上生成异常"""
        if self.method == 'cutpaste':
            return self._cutpaste(image)
        elif self.method == 'copy_paste':
            return self._copy_paste(image)
        elif self.method == 'noise_injection':
            return self._add_noise(image)
        elif self.method == 'blur_sharpness':
            return self._blur_sharpness(image)
        return image

    def _cutpaste(self, image: np.ndarray) -> np.ndarray:
        """CutPaste实现"""
        h, w, c = image.shape
        crop_h = int(h * np.random.choice(self.crop_sizes))
        crop_w = int(w * np.random.choice(self.crop_sizes))
        y1 = np.random.randint(0, h - crop_h)
        x1 = np.random.randint(0, w - crop_w)
        crop = image[y1:y1+crop_h, x1:x1+crop_w].copy()

        scale = np.random.choice(self.crop_scales)
        if scale != 1.0:
            crop = cv2.resize(crop, None, fx=scale, fy=scale)

        y2 = np.random.randint(0, h - crop_h)
        x2 = np.random.randint(0, w - crop_w)

        result = image.copy()
        alpha = 0.8
        result[y2:y2+crop_h, x2:x2+crop_w] = (
            alpha * crop + (1 - alpha) * result[y2:y2+crop_h, x2:x2+crop_w]
        )
        return result

    def _copy_paste(self, image: np.ndarray) -> np.ndarray:
        """Copy-Paste实现"""
        n_anomalies = np.random.randint(1, 4)
        result = image.copy()
        for _ in range(n_anomalies):
            size = np.random.uniform(0.05, 0.2)
            crop_h = int(image.shape[0] * size)
            crop_w = int(image.shape[1] * size)
            y1 = np.random.randint(0, image.shape[0] - crop_h)
            x1 = np.random.randint(0, image.shape[1] - crop_w)
            crop = image[y1:y1+crop_h, x1:x1+crop_w].copy()
            result = self._poisson_blend(crop, result, (y1, x1))
        return result

    def _add_noise(self, image: np.ndarray) -> np.ndarray:
        """添加噪声异常"""
        noise_type = np.random.choice(['gaussian', 'salt', 'speckle'])
        noise_level = np.random.uniform(0.05, 0.3)

        if noise_type == 'gaussian':
            noise = np.random.normal(0, noise_level * 255, image.shape)
            return np.clip(image + noise, 0, 255).astype(np.uint8)
        elif noise_type == 'salt':
            result = image.copy()
            n_pixels = int(image.size * noise_level * 0.5)
            coords = [np.random.randint(0, s, n_pixels) for s in image.shape[:2]]
            result[coords[0], coords[1]] = 255
            return result
        return image

    def _blur_sharpness(self, image: np.ndarray) -> np.ndarray:
        """模糊/清晰度异常"""
        operation = np.random.choice(['blur', 'sharpen'])
        if operation == 'blur':
            ksize = np.random.choice([3, 5, 7, 9])
            return cv2.GaussianBlur(image, (ksize, ksize), 0)
        else:
            kernel = np.array([[0, -1, 0], [-1, 5], [0, -1, 0]])
            return cv2.filter2D(image, -1, kernel)

    def _poisson_blend(self, src: np.ndarray, dst: np.ndarray, offset: tuple) -> np.ndarray:
        """泊松融合 (简化版)"""
        y, x = offset
        h, w = src.shape[:2]
        y = max(0, min(y, dst.shape[0] - h))
        x = max(0, min(x, dst.shape[1] - w))
        mask = np.zeros((h, w, 1), dtype=np.float32)
        cv2.circle(mask, (w//2, h//2), min(w, h)//2, (1,), -1)
        result = dst.copy()
        region = result[y:y+h, x:x+w].astype(np.float32)
        blended = src.astype(np.float32) * mask + region * (1 - mask)
        result[y:y+h, x:x+w] = blended.astype(np.uint8)
        return result


class AugmentedDataset:
    """
    增强数据集wrapper - 训练时动态生成异常
    """

    def __init__(self, base_dataset, augmentor: AnomalyAugmentor,
                 anomaly_ratio: float = 0.5):
        self.base_dataset = base_dataset
        self.augmentor = augmentor
        self.anomaly_ratio = anomaly_ratio

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]

        # 正常样本考虑增强
        if label != 'anomaly':
            if np.random.random() < self.anomaly_ratio:
                image = self.augmentor.generate_anomaly(image)

        return image, label
```

---

## 四、RL控制器设计

### 4.1 控制器架构

```python
class NASController(nn.Module):
    """
    RNN控制器 - 用于生成架构编码
    基于NASNet风格的强化学习搜索
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.get('controller_hidden', 100)
        self.num_layers = config.get('controller_layers', 1)
        self.num_operations = config.get('num_operations', 12)

        # 嵌入层
        self.embeddings = nn.ModuleList([
            nn.Embedding(config['num_layers'] + 1, self.hidden_dim),
            nn.Embedding(config['num_operations'], self.hidden_dim),
        ])

        # LSTM控制器
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            bias=True
        )

        # 多头输出层
        self.decoders = nn.ModuleList([
            nn.Linear(self.hidden_dim, len(ENCODER_OPTIONS['backbone'])),
            nn.Linear(self.hidden_dim, len(FEATURE_OPTIONS['levels'])),
            nn.Linear(self.hidden_dim, len(HEAD_OPTIONS['method'])),
            nn.Linear(self.hidden_dim, len(HEAD_OPTIONS['memory_bank']['size'])),
        ])

    def forward(self, inputs, hidden=None):
        """生成架构概率分布"""
        embedded = self.embeddings[0](inputs)
        output, hidden = self.lstm(embedded, hidden)

        logits = [decoder(output) for decoder in self.decoders]
        return logits, hidden
```

### 4.2 奖励函数设计

```python
class RewardFunction:
    """
    多目标奖励函数 - 针对边缘部署优化
    综合考虑检测性能和推理效率
    """
    def __init__(self, weights=None):
        self.weights = weights or {
            'auroc': 0.4,       # 检测性能权重 (边缘设备需要平衡)
            'f1': 0.15,         # F1分数权重
            'params': 0.15,     # 模型大小惩罚 (边缘关键)
            'latency': 0.20,    # 推理延迟惩罚 (边缘关键)
            'memory': 0.10      # 内存使用惩罚
        }

    def compute(self, metrics: Dict) -> float:
        """
        计算综合奖励分数

        Args:
            metrics: {'auroc': float, 'f1': float, 'params': int, 'latency': float, 'memory': float}
        """
        reward = 0.0

        # 检测性能奖励 (越大越好)
        reward += self.weights['auroc'] * metrics['auroc']
        reward += self.weights['f1'] * metrics['f1']

        # 模型效率惩罚 (越小越好) - 边缘部署关键
        params_score = 1.0 / (1.0 + np.log1p(metrics['params'] / 1e6))
        reward += self.weights['params'] * params_score

        latency_score = 1.0 / (1.0 + metrics['latency'] / 50)  # 50ms基准
        reward += self.weights['latency'] * latency_score

        memory_score = 1.0 / (1.0 + metrics['memory'] / 1024)
        reward += self.weights['memory'] * memory_score

        return reward

    def pareto_filter(self, results: List[Dict]) -> List[Dict]:
        """帕累托过滤 - 保留非支配解"""
        pareto_results = []
        for r in results:
            dominated = False
            for other in results:
                if other != r and self._dominates(other, r):
                    dominated = True
                    break
            if not dominated:
                pareto_results.append(r)
        return pareto_results
```

### 4.3 多维度架构评估器 (Multi-Dimension Evaluator)

```python
class ArchitectureSimilarityEncoder:
    """
    架构相似度编码器

    将神经网络架构编码为向量，用于计算与专家设计的相似度

    方法:
    1. 操作分布编码: 统计各类型操作(Conv, Pool, Attention)的比例
    2. 拓扑结构编码: DAG结构的图嵌入
    3. 参数分布编码: 各层参数量的分布
    """

    def __init__(self, expert_architectures: List[Dict]):
        """
        Args:
            expert_architectures: 专家设计的架构列表 (PatchCore, PaDiM等)
        """
        self.expert_embeddings = [self.encode(arch) for arch in expert_architectures]

    def encode(self, architecture: Dict) -> np.ndarray:
        """将架构编码为向量"""
        features = []

        # 1. 操作分布编码 (8维)
        layer_types = self._count_layer_types(architecture)
        features.extend([
            layer_types.get('conv', 0) / max(1, sum(layer_types.values())),
            layer_types.get('dense', 0) / max(1, sum(layer_types.values())),
            layer_types.get('pooling', 0) / max(1, sum(layer_types.values())),
            layer_types.get('attention', 0) / max(1, sum(layer_types.values())),
            layer_types.get('bn', 0) / max(1, sum(layer_types.values())),
            layer_types.get('dropout', 0) / max(1, sum(layer_types.values())),
            layer_types.get('skip', 0) / max(1, sum(layer_types.values())),
            architecture.get('num_layers', 1) / 50.0  # 归一化深度
        ])

        # 2. 特征层级选择 (4维)
        levels = architecture.get('feature_levels', [2])
        features.extend([
            1 if 1 in levels else 0,
            1 if 2 in levels else 0,
            1 if 3 in levels else 0,
            1 if 4 in levels else 0
        ])

        # 3. 检测头类型 (one-hot 4维)
        head_type = architecture.get('detection_head', {}).get('method', 'memory_bank')
        head_map = {'memory_bank': 0, 'distribution': 1, 'student_teacher': 2, 'contrastive': 3}
        head_onehot = [0] * 4
        head_onehot[head_map.get(head_type, 0)] = 1
        features.extend(head_onehot)

        # 4. 注意力模块 (one-hot 5维)
        attn = architecture.get('attention_module', 'none')
        attn_map = {'none': 0, 'SE': 1, 'CBAM': 2, 'ECA': 3, 'TripletAttention': 4}
        attn_onehot = [0] * 5
        attn_onehot[attn_map.get(attn, 0)] = 1
        features.extend(attn_onehot)

        # 5. 记忆库配置 (3维)
        mb = architecture.get('memory_bank', {})
        features.extend([
            min(mb.get('size', 1000) / 10000, 1.0),
            1 if mb.get('sampling') == 'kcenter' else 0,
            1 if mb.get('reduction') == 'pca' else 0
        ])

        return np.array(features)

    def _count_layer_types(self, architecture: Dict) -> Dict[str, int]:
        """统计各类型层的数量"""
        counts = {}
        layers = architecture.get('layers', [])

        # 兼容不同格式
        if isinstance(layers, dict):
            layers = list(layers.values())

        for layer in layers:
            layer_type = layer.get('type', layer.get('name', 'unknown'))
            layer_type = layer_type.lower().split('_')[0]
            counts[layer_type] = counts.get(layer_type, 0) + 1

        return counts

    def compute_expert_similarity(self, architecture: Dict) -> float:
        """
        计算与专家设计的相似度

        Returns:
            相似度分数 [0, 1]，1表示与专家设计高度一致
        """
        arch_embedding = self.encode(architecture)

        # 计算与所有专家设计的余弦相似度
        similarities = []
        for expert_emb in self.expert_embeddings:
            sim = cosine_similarity([arch_embedding], [expert_emb])[0][0]
            similarities.append(sim)

        # 返回最高相似度
        return max(similarities)


class MultiDimensionEvaluator:
    """
    多维度架构评估器

    评估维度:
    1. 准确度 (Accuracy) - AUROC, F1, AUPRO
    2. 效率 (Efficiency) - 参数量, FLOPs, 延迟
    3. 专家相似度 (Expert Alignment) - 与PatchCore/PaDiM等设计的相似度
    4. 泛化能力 (Generalization) - 跨数据集表现
    """

    def __init__(self, expert_architectures: List[Dict], device: str = 'cuda'):
        self.device = device

        # 准确度评估器
        self.accuracy_evaluator = AccuracyEvaluator()

        # 效率评估器
        self.efficiency_evaluator = EfficiencyEvaluator()

        # 专家相似度评估器
        self.similarity_encoder = ArchitectureSimilarityEncoder(expert_architectures)

        # 评估权重
        self.weights = {
            'accuracy': 0.40,      # 检测性能
            'efficiency': 0.25,   # 推理效率
            'expert_similarity': 0.20,  # 专家对齐
            'generalization': 0.15  # 泛化能力
        }

    def evaluate(self, architecture: Dict, test_loader: DataLoader,
                 validation_loader: DataLoader = None) -> Dict:
        """
        综合评估架构

        Returns:
            包含所有维度分数的评估结果
        """
        results = {}

        # 1. 准确度评估
        accuracy_metrics = self.accuracy_evaluator.evaluate(architecture, test_loader)
        results['accuracy'] = {
            'auroc': accuracy_metrics['auroc'],
            'f1': accuracy_metrics['f1'],
            'aupro': accuracy_metrics.get('aupro', 0),
            'score': self._normalize_accuracy(accuracy_metrics['auroc'])
        }

        # 2. 效率评估
        efficiency_metrics = self.efficiency_evaluator.evaluate(architecture)
        results['efficiency'] = {
            'params': efficiency_metrics['params'],
            'flops': efficiency_metrics['flops'],
            'latency_ms': efficiency_metrics['latency'],
            'score': self._normalize_efficiency(efficiency_metrics)
        }

        # 3. 专家相似度评估
        similarity_score = self.similarity_encoder.compute_expert_similarity(architecture)
        results['expert_similarity'] = {
            'raw_score': similarity_score,
            'score': similarity_score  # 已在[0,1]范围
        }

        # 4. 泛化能力评估 (如果提供了验证集)
        if validation_loader is not None:
            generalization_metrics = self.accuracy_evaluator.evaluate(architecture, validation_loader)
            results['generalization'] = {
                'validation_auroc': generalization_metrics['auroc'],
                'score': self._normalize_generalization(accuracy_metrics['auroc'],
                                                          generalization_metrics['auroc'])
            }
        else:
            results['generalization'] = {
                'validation_auroc': None,
                'score': 0.5  # 默认中等分数
            }

        # 5. 综合分数
        results['overall_score'] = (
            self.weights['accuracy'] * results['accuracy']['score'] +
            self.weights['efficiency'] * results['efficiency']['score'] +
            self.weights['expert_similarity'] * results['expert_similarity']['score'] +
            self.weights['generalization'] * results['generalization']['score']
        )

        # 6. 详细报告
        results['report'] = self._generate_report(results)

        return results

    def _normalize_accuracy(self, auroc: float) -> float:
        """将AUROC映射到[0,1]分数"""
        # AUROC 0.5是随机猜测，1.0是完美
        return max(0.0, min(1.0, (auroc - 0.5) * 2))

    def _normalize_efficiency(self, metrics: Dict) -> float:
        """综合效率分数"""
        # 参数量分数 (越小越好)
        params_score = 1.0 / (1.0 + np.log1p(metrics['params'] / 1e6))

        # 延迟分数 (越小越好)
        latency_score = 1.0 / (1.0 + metrics['latency'] / 50)

        # FLOPs分数 (越小越好)
        flops_score = 1.0 / (1.0 + np.log1p(metrics['flops'] / 1e9))

        return (params_score * 0.3 + latency_score * 0.5 + flops_score * 0.2)

    def _normalize_generalization(self, test_auroc: float, val_auroc: float) -> float:
        """泛化能力分数 - 测试集与验证集差异"""
        gap = abs(test_auroc - val_auroc)
        # 差异越小分数越高
        return max(0.0, 1.0 - gap * 2)

    def _generate_report(self, results: Dict) -> str:
        """生成人类可读的评估报告"""
        return f"""
        ===== 架构评估报告 =====
        准确度 (AUROC): {results['accuracy']['auroc']:.4f}
        推理延迟: {results['efficiency']['latency_ms']:.2f}ms
        参数量: {results['efficiency']['params']/1e6:.2f}M
        专家相似度: {results['expert_similarity']['raw_score']:.4f}
        综合分数: {results['overall_score']:.4f}
        ========================
        """


class ExpertArchitectureLibrary:
    """
    专家架构库 - 存储已验证的专家设计

    来源:
    - PatchCore (CVPR 2022)
    - PaDiM (CVPR 2021)
    - Student-Teacher (BMVC 2021)
    - RD4AD (CVPR 2021)
    """

    def __init__(self):
        self.expert_architectures = self._load_expert_architectures()

    def _load_expert_architectures(self) -> List[Dict]:
        """加载专家架构定义"""
        return [
            # PatchCore 风格
            {
                'name': 'PatchCore',
                'backbone': 'WideResNet50',
                'feature_levels': [2, 3],
                'detection_head': {'method': 'memory_bank'},
                'attention_module': 'none',
                'sampling': 'kcenter',
                'k': 9
            },
            # PaDiM 风格
            {
                'name': 'PaDiM',
                'backbone': 'ResNet50',
                'feature_levels': [1, 2, 3],
                'detection_head': {'method': 'distribution'},
                'attention_module': 'none',
                'reduction': 'pca',
                'dim_reduction': 100
            },
            # Student-Teacher 风格
            {
                'name': 'StudentTeacher',
                'backbone': 'EfficientNet-B5',
                'feature_levels': [2, 3, 4],
                'detection_head': {'method': 'student_teacher'},
                'attention_module': 'CBAM',
                'teacher_pretrained': True
            }
        ]

    def get_architecture(self, name: str) -> Dict:
        """获取指定专家架构"""
        for arch in self.expert_architectures:
            if arch['name'].lower() == name.lower():
                return arch
        return None

    def list_architectures(self) -> List[str]:
        """列出所有专家架构"""
        return [arch['name'] for arch in self.expert_architectures]
```

---

## 五、LLM Agent设计

### 5.1 LLM Agent架构

```python
class LLMArchitectureAgent:
    """
    LLM架构设计Agent
    负责分析搜索结果、提出架构改进建议
    """

    def __init__(self, llm_client, rag_system):
        self.llm = llm_client  # MiniMax M2.1 / OpenAI / Claude
        self.rag = rag_system

    def analyze_search_results(self, search_history: List[Dict]) -> Dict:
        """
        分析搜索历史，提供架构设计洞察
        """
        # 1. RAG检索相关案例
        context = self.rag.retrieve(
            query=f"architecture optimization industrial anomaly detection",
            top_k=5
        )

        # 2. 构建Prompt
        prompt = self._build_analysis_prompt(search_history, context)

        # 3. LLM分析
        analysis = self.llm.generate(prompt)

        return {
            'insights': analysis['suggestions'],
            'promising_architectures': analysis['recommendations'],
            'failure_patterns': analysis['patterns_to_avoid']
        }

    def suggest_architecture(self, task_spec: Dict) -> Dict:
        """
        根据任务规格建议初始架构
        """
        prompt = f"""
        为工业异常检测任务设计架构。

        任务规格:
        - 数据集: {task_spec.get('dataset', 'MVTec AD')}
        - 类别数: {task_spec.get('num_classes', 2)}
        - 部署平台: {task_spec.get('platform', 'GPU')}
        - 延迟约束: {task_spec.get('latency_constraint', '100ms')}
        - 精度目标: {task_spec.get('accuracy_target', 'AUROC > 0.95')}

        请输出:
        1. 推荐架构配置
        2. 设计理由
        3. 预期性能范围
        """

        return self.llm.generate(prompt)
```

### 5.1 LLM Agent 设计 (MiniMax M2.1)

```python
# LLM客户端统一接口
class LLMClientBase:
    """LLM客户端基类 - 支持多种后端"""
    def generate(self, prompt: str, **kwargs) -> Dict:
        raise NotImplementedError


class MiniMaxM2Client(LLMClientBase):
    """
    MiniMax M2.1 API 客户端

    优势:
    - 国内访问延迟低
    - 支持长上下文 (128K)
    - 成本低于GPT-4
    - 中文优化
    """
    def __init__(self, api_key: str, base_url: str = "https://api.minimax.chat/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = "MiniMax-M2.1"  # 或 "MiniMax-M2.1-8k" 等变体

    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> Dict:
        import requests

        response = requests.post(
            f"{self.base_url}/text/chatcompletion_v2",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        )

        result = response.json()
        return {
            'content': result['choices'][0]['message']['content'],
            'usage': result['usage'],
            'model': self.model
        }


class LLMClientFactory:
    """LLM客户端工厂 - 统一接口"""
    @staticmethod
    def create(config: Dict) -> LLMClientBase:
        provider = config.get('provider', 'minimax')

        if provider == 'minimax':
            return MiniMaxM2Client(
                api_key=config['api_key'],
                base_url=config.get('base_url', 'https://api.minimax.chat/v1')
            )
        elif provider == 'openai':
            return OpenAIClient(api_key=config['api_key'], model=config.get('model', 'gpt-4'))
        elif provider == 'anthropic':
            return AnthropicClient(api_key=config['api_key'], model=config.get('model', 'claude-3'))
        else:
            raise ValueError(f"Unknown provider: {provider}")


class LLMArchitectureAgent:
    """
    LLM架构设计Agent
    负责分析搜索结果、提出架构改进建议
    使用MiniMax M2.1提供低成本高质量的推理
    """

    def __init__(self, llm_client: LLMClientBase, rag_system):
        self.llm = llm_client
        self.rag = rag_system

    def analyze_search_results(self, search_history: List[Dict]) -> Dict:
        """
        分析搜索历史，提供架构设计洞察
        """
        # 1. RAG检索相关案例
        context = self.rag.retrieve(
            query=f"architecture optimization industrial anomaly detection",
            top_k=5
        )

        # 2. 构建Prompt
        prompt = self._build_analysis_prompt(search_history, context)

        # 3. LLM分析
        analysis = self.llm.generate(prompt)

        return {
            'insights': analysis['content'],
            'promising_architectures': [],
            'failure_patterns': []
        }

    def suggest_architecture(self, task_spec: Dict) -> Dict:
        """
        根据任务规格建议初始架构
        """
        prompt = f"""
        为工业异常检测任务设计架构。

        任务规格:
        - 数据集: {task_spec.get('dataset', 'MVTec AD')}
        - 类别数: {task_spec.get('num_classes', 2)}
        - 部署平台: {task_spec.get('platform', 'GPU')}
        - 延迟约束: {task_spec.get('latency_constraint', '100ms')}
        - 精度目标: {task_spec.get('accuracy_target', 'AUROC > 0.95')}

        请输出JSON格式:
        {{
            "backbone": "推荐的网络骨架",
            "feature_levels": ["使用的特征层级"],
            "detection_head": "检测头类型",
            "attention_module": "注意力模块",
            "reasoning": "设计理由"
        }}
        """

        return self.llm.generate(prompt)
```

### 5.2 RAG系统设计

```python
class IndustrialAnomalyRAG:
    """
    工业异常检测RAG系统
    检索相关论文、架构案例、性能基准
    """

    def __init__(self, vector_db_path: str):
        self.vector_store = load_vector_store(vector_db_path)
        self.document_store = load_document_store()

        # 文档来源
        self.sources = [
            'papers/ patchcore_paper.pdf',
            'papers/_padim_paper.pdf',
            'papers/nas_survey.pdf',
            'benchmarks/mvtec_results.json',
            'architecture_cases/resnet_cases.json',
            'deployment_cases/edge_deployment.json'
        ]

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """检索最相关的文档"""
        # 1. 向量化查询
        query_embedding = self._encode(query)

        # 2. 向量检索
        results = self.vector_store.search(query_embedding, top_k=top_k)

        # 3. 获取原文
        enriched_results = []
        for r in results:
            doc = self.document_store.get(r['doc_id'])
            enriched_results.append({
                'content': doc['content'],
                'source': doc['source'],
                'metadata': r['metadata'],
                'relevance_score': r['score']
            })

        return enriched_results

    def add_architecture_case(self, architecture: Dict, performance: Dict):
        """添加新的架构案例到知识库"""
        # 1. 提取架构特征
        features = self._extract_features(architecture)

        # 2. 转换为向量
        embedding = self._encode(str(features))

        # 3. 存储
        self.vector_store.add(
            id=str(uuid.uuid4()),
            embedding=embedding,
            metadata={
                'performance': performance,
                'architecture': architecture,
                'timestamp': datetime.now().isoformat()
            }
        )
```

### 5.3 经验积累与持续学习 (Experience Accumulation)

```python
class ExperienceAccumulator:
    """
    经验积累器 - 将搜索经验持续转化为可复用知识

    核心思想:
    1. 每次搜索评估后提取经验（成功/失败模式）
    2. 自动生成设计建议和陷阱提醒
    3. 定期汇总分析，提炼知识沉淀
    4. 供LLM Agent检索和参考
    """

    def __init__(self, rag_system: IndustrialAnomalyRAG):
        self.rag = rag_system
        self.experience_buffer = []  # 短期经验缓冲
        self.long_term_patterns = {}  # 长期模式总结

    def record_trial(self, architecture: Dict, metrics: Dict, search_context: Dict):
        """
        记录一次搜索尝试

        Args:
            architecture: 架构配置
            metrics: 性能指标 (auroc, latency, params等)
            search_context: 搜索上下文 (数据集、目标设备等)
        """
        experience = {
            'id': str(uuid.uuid4()),
            'architecture': architecture,
            'metrics': metrics,
            'context': search_context,
            'timestamp': datetime.now().isoformat(),
            'status': self._classify_status(metrics)
        }

        self.experience_buffer.append(experience)

        # 定期同步到RAG知识库 (每N次尝试)
        if len(self.experience_buffer) >= self.sync_interval:
            self._sync_to_knowledge_base()

    def _classify_status(self, metrics: Dict) -> str:
        """分类搜索结果状态"""
        auroc = metrics.get('auroc', 0)
        latency = metrics.get('latency', float('inf'))

        if auroc >= 0.95 and latency < 50:
            return 'excellent'  # 优秀架构
        elif auroc >= 0.90:
            return 'good'  # 良好架构
        elif auroc >= 0.70:
            return 'acceptable'  # 可接受
        elif auroc < 0.50:
            return 'failed'  # 失败
        else:
            return 'partial'  # 部分失败

    def _sync_to_knowledge_base(self):
        """同步经验到RAG知识库"""
        # 1. 分析缓冲期内的成功模式
        success_patterns = self._analyze_patterns('excellent', 'good')

        # 2. 分析失败模式
        failure_patterns = self._analyze_patterns('failed')

        # 3. 生成设计建议
        suggestions = self._generate_suggestions(success_patterns, failure_patterns)

        # 4. 存储到RAG
        self._store_patterns_to_rag(success_patterns, failure_patterns, suggestions)

        # 5. 清空缓冲
        self.experience_buffer = []

    def _analyze_patterns(self, *statuses) -> Dict:
        """分析特定状态的模式"""
        relevant = [e for e in self.experience_buffer if e['status'] in statuses]

        patterns = {
            'count': len(relevant),
            'avg_auroc': np.mean([e['metrics'].get('auroc', 0) for e in relevant]),
            'common_backbones': self._most_common(relevant, 'encoder.backbone'),
            'common_heads': self._most_common(relevant, 'detection_head.method'),
            'feature_combinations': self._extract_feature_combos(relevant),
            'performance_range': {
                'min': min(e['metrics'].get('auroc', 0) for e in relevant),
                'max': max(e['metrics'].get('auroc', 0) for e in relevant)
            }
        }

        return patterns

    def _generate_suggestions(self, success_patterns: Dict, failure_patterns: Dict) -> List[str]:
        """生成设计建议"""
        suggestions = []

        # 基于成功模式
        if success_patterns['count'] > 0:
            if 'ResNet50' in success_patterns.get('common_backbones', []):
                suggestions.append({
                    'type': 'recommendation',
                    'content': 'ResNet50 backbone在本任务表现稳定，推荐作为首选',
                    'confidence': 0.8
                })

        # 基于失败模式
        if failure_patterns['count'] > 0:
            suggestions.append({
                'type': 'warning',
                'content': f"检测到{failure_patterns['count']}次失败尝试，避免使用高延迟架构组合",
                'confidence': 0.9
            })

        return suggestions

    def _store_patterns_to_rag(self, success: Dict, failure: Dict, suggestions: List):
        """存储分析结果到RAG"""
        # 生成自然语言描述
        summary = f"""
        搜索经验总结:
        - 成功尝试: {success.get('count', 0)} 次, 平均AUROC: {success.get('avg_auroc', 0):.3f}
        - 常用backbone: {success.get('common_backbones', [])}
        - 常用检测头: {success.get('common_heads', [])}
        - 失败尝试: {failure.get('count', 0)} 次
        """

        # 存储到知识库
        self.rag.add_experience_summary(
            summary=summary,
            suggestions=suggestions,
            patterns={'success': success, 'failure': failure}
        )

    def get_learned_insights(self) -> Dict:
        """获取累积的学习洞察"""
        return self.long_term_patterns
```

### 5.4 长期知识蒸馏 (Knowledge Distillation to RAG)

```python
class KnowledgeDistiller:
    """
    知识蒸馏器 - 将搜索经验提炼为高质量知识

    定期运行:
    1. 汇总大量搜索结果
    2. 使用LLM提炼洞察
    3. 更新RAG中的结构化知识
    """

    def __init__(self, llm_agent, rag_system):
        self.llm = llm_agent
        self.rag = rag_system
        self.min_trials_for_distillation = 50  # 最少50次尝试后才蒸馏

    def run_distillation(self, all_trials: List[Dict]):
        """执行知识蒸馏"""
        if len(all_trials) < self.min_trials_for_distillation:
            return  # 样本不足

        # 1. 按性能分组
        top_10 = sorted(all_trials, key=lambda x: x['metrics']['auroc'], reverse=True)[:10]
        bottom_10 = sorted(all_trials, key=lambda x: x['metrics']['auroc'])[:10]

        # 2. LLM分析
        prompt = f"""
        基于以下搜索数据，提炼工业异常检测架构设计的关键洞察:

        最佳10个架构:
        {self._format_architectures(top_10)}

        最差10个架构:
        {self._format_architectures(bottom_10)}

        请分析:
        1. 成功架构的共同特征
        2. 失败架构的典型问题
        3. 针对边缘部署的具体建议
        4. 未来搜索的探索方向
        """

        insights = self.llm.generate(prompt)

        # 3. 更新RAG知识库
        self.rag.add_distilled_knowledge(insights)

        return insights
```

### 5.5 经验查询与利用

```python
class ExperienceAwareSearch:
    """
    经验感知搜索器 - 利用历史经验指导搜索

    策略:
    1. 检索相似任务的历史经验
    2. 优先尝试成功的设计模式
    3. 避免已知的失败组合
    """

    def __init__(self, rag_system, search_space):
        self.rag = rag_system
        self.search_space = search_space

    def get_guided_suggestions(self, task_spec: Dict) -> List[Dict]:
        """获取经验指导的架构建议"""
        # 1. RAG检索相似经验
        similar_experiences = self.rag.retrieve(
            query=f"anomaly detection {task_spec.get('dataset')} {task_spec.get('target_device')}",
            top_k=10
        )

        # 2. 提取成功架构模式
        successful_patterns = [
            e for e in similar_experiences
            if e['metadata'].get('performance', {}).get('auroc', 0) > 0.90
        ]

        # 3. 生成架构建议
        suggestions = []
        for pattern in successful_patterns[:3]:
            arch = pattern['metadata']['architecture']
            suggestions.append({
                'architecture': arch,
                'source': 'historical_success',
                'confidence': pattern['relevance_score']
            })

        return suggestions

    def get_avoidance_list(self) -> List[Dict]:
        """获取应避免的架构组合"""
        failures = self.rag.retrieve(
            query="failed architecture combination low performance",
            top_k=20
        )
        return [f['metadata']['architecture'] for f in failures]
```

---

## 六、效率优化策略

### 6.0 多GPU成本优化 (并行智算云)

```python
# 成本优化配置 - 并行智算云 RTX 4090/5090
COST_OPTIMIZATION = {
    'max_search_budget': {
        'total_hours': 72,        # 最大搜索时间
        'max_trials': 500,         # 最大搜索次数 (多GPU加速)
        'gpu_hours_limit': 200     # GPU小时限制
    },
    'cost_per_trial': {
        'low_fidelity': 0.005,    # ¥ - 快速筛选 (1 epoch, 1 GPU)
        'medium_fidelity': 0.02,   # ¥ - 中等评估 (10 epochs, 2 GPUs)
        'high_fidelity': 0.10      # ¥¥ - 完整评估 (50 epochs, 4 GPUs)
    },
    'estimated_total_cost': '¥100-500 (使用代金券)',
    'recommended_config': '4x RTX 4090 或 8x RTX 3090',
    'gpu_speedup': {
        '1_gpu': 1.0,
        '2_gpu': 1.8x,
        '4_gpu': 3.2x,
        '8_gpu': 5.5x
    }
}


class CostAwareSearchScheduler:
    """
    成本感知搜索调度器 - 云端租用优化
    """
    def __init__(self, budget: float = 100.0):
        self.budget = budget
        self.spent = 0.0

    def select_trial_strategy(self) -> str:
        remaining = self.budget - self.spent
        if remaining < 10:
            return 'ultra_fast'
        elif remaining < 30:
            return 'balanced'
        else:
            return 'thorough'

    def track_cost(self, trial_cost: float):
        self.spent += trial_cost
        if self.spent >= self.budget:
            raise BudgetExceededError(f"预算已用完: ${self.spent:.2f}")
```

### 6.1 权重共享超网络

```python
class IndustrialAnomalySuperNet(nn.Module):
    """
    超网络 - 支持权重共享的NAS

    特点:
    - 所有候选架构共享同一组权重
    - 只需训练一次超网络
    - 大幅减少搜索成本
    """

    def __init__(self, config: SearchSpaceConfig):
        super().__init__()

        # 共享的编码器backbone
        self.encoder = create_shared_encoder(
            in_channels=config.input_channels,
            base_channels=config.base_channels
        )

        # 候选操作的超网络变体
        self.operation_supernet = nn.ModuleDict({
            'conv_3x3': self._create_conv_block(3),
            'conv_5x5': self._create_conv_block(5),
            'conv_7x7': self._create_conv_block(7),
            'depthwise_3x3': self._create_dw_block(3),
            'se_block': SEBlock(),
            'cbam_block': CBAMBlock(),
            'skip': nn.Identity(),
            'pool_avg': nn.AdaptiveAvgPool2d(1),
            'pool_max': nn.AdaptiveMaxPool2d(1)
        })

        # 混合权重 (用于DARTS-style搜索)
        self.alpha_ops = nn.Parameter(
            torch.randn(len(self.operation_supernet))
        )

    def forward(self, x, architecture_sample: Dict):
        """前向传播"""
        # 编码器
        features = self.encoder(x)

        # 根据架构采样选择操作
        selected_ops = architecture_sample.get('operations', [])

        outputs = []
        for i, op_name in enumerate(selected_ops):
            if op_name in self.operation_supernet:
                outputs.append(self.operation_supernet[op_name](features[i]))

        return outputs
```

### 6.2 多保真度评估

```python
class MultiFidelityEvaluator:
    """
    多保真度NAS评估器

    策略:
    - Level 1: 1 epoch, 10% 数据 (快速筛选)
    - Level 2: 10 epochs, 30% 数据 (中间筛选)
    - Level 3: 50 epochs, 100% 数据 (精细评估)
    """

    def __init__(self, train_dataset, val_dataset):
        self.datasets = {
            'low': (train_dataset.sample(0.1), val_dataset.sample(0.1)),
            'medium': (train_dataset.sample(0.3), val_dataset.sample(0.3)),
            'high': (train_dataset, val_dataset)
        }

        self.fidelity_thresholds = {
            'low': 0.5,      # AUROC < 0.5 停止
            'medium': 0.7,   # AUROC < 0.7 停止
            'high': 0.0      # 完整评估
        }

    def evaluate(self, architecture: Dict, early_stop_score: float = None) -> Dict:
        """评估架构"""
        for fidelity_level in ['low', 'medium', 'high']:
            # 获取数据集
            train_data, val_data = self.datasets[fidelity_level]

            # 训练
            model = build_model(architecture)
            train_metrics = self._train_model(model, train_data, epochs=self.epochs[fidelity_level])

            # 验证
            val_metrics = self._validate(model, val_data)

            # 检查是否应该停止
            if val_metrics['auroc'] < self.fidelity_thresholds[fidelity_level]:
                return {
                    'auroc': val_metrics['auroc'],
                    'fidelity': fidelity_level,
                    'stopped_early': True
                }

        return {
            'auroc': val_metrics['auroc'],
            'fidelity': 'high',
            'stopped_early': False,
            'params': model.count_parameters(),
            'latency': self._measure_latency(model)
        }
```

### 6.3 性能预测器

```python
class PerformancePredictor:
    """
    性能预测器 - 基于架构特征预测最终性能

    避免完整训练，快速评估架构潜力
    """

    def __init__(self):
        self.model = self._build_predictor()

    def _build_predictor(self):
        """构建预测器网络"""
        return nn.Sequential(
            nn.Linear(self._get_feature_dim(), 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 预测AUROC
        )

    def extract_features(self, architecture: Dict) -> np.ndarray:
        """提取架构特征"""
        features = []

        # 参数量特征
        total_params = self._count_params(architecture)
        features.extend([
            np.log1p(total_params),
            np.log1p(total_params) ** 2,
            total_params > 1e6
        ])

        # FLOPs估计
        flops = self._estimate_flops(architecture)
        features.extend([np.log1p(flops), flops > 1e9])

        # 深度特征
        features.append(architecture.get('num_layers', 1))

        # 操作类型分布
        op_counts = architecture.get('operation_counts', {})
        features.extend([
            op_counts.get('conv', 0),
            op_counts.get('attention', 0),
            op_counts.get('pooling', 0)
        ])

        return np.array(features)

    def predict(self, architecture: Dict) -> float:
        """预测性能"""
        features = self.extract_features(architecture)
        with torch.no_grad():
            return float(self.model(torch.FloatTensor(features)))
```

### 6.4 边缘设备部署优化 (嵌入式)

```python
class EdgeDeploymentOptimizer:
    """
    边缘设备优化器 - 针对嵌入式部署
    目标设备: NVIDIA Jetson, Raspberry Pi, Intel Movidius
    """
    def __init__(self, target_device: str = 'jetson_nano'):
        self.target_device = target_device
        self.constraints = {
            'jetson_nano': {
                'max_params': 10e6,     # 10M参数量
                'max_latency': 50,       # ms
                'max_memory': 2048,      # MB
                'precision': 'int8'
            },
            'jetson_xavier': {
                'max_params': 50e6,
                'max_latency': 30,
                'max_memory': 4096,
                'precision': 'int8'
            },
            'raspberry_pi_4': {
                'max_params': 5e6,
                'max_latency': 100,
                'max_memory': 1024,
                'precision': 'fp32'
            }
        }

    def apply_constraints(self, architecture: Dict) -> Dict:
        """应用部署约束"""
        constraints = self.constraints[self.target_device]

        # 参数量限制
        if self._count_params(architecture) > constraints['max_params']:
            architecture = self._prune_architecture(architecture)

        return architecture

    def quantize(self, model: nn.Module) -> nn.Module:
        """INT8量化"""
        from torch.quantization import quantize_dynamic
        return quantize_dynamic(model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)

    def export_for_device(self, model: nn.Module, output_path: str):
        """导出到目标设备格式"""
        if self.target_device.startswith('jetson'):
            # TensorRT导出
            import tensorrt as trt
            self._export_tensorrt(model, output_path)
        else:
            # TFLite导出
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            with open(output_path, 'wb') as f:
                f.write(converter.convert())
```

### 6.5 模型导出格式

```python
EXPORT_FORMATS = {
    'tensorrt': {
        'target': ['Jetson Nano', 'Jetson Xavier', 'Jetson Orin'],
        'precision': ['fp32', 'fp16', 'int8'],
        'speedup': '2-10x'
    },
    'tflite': {
        'target': ['Raspberry Pi', 'Edge TPU', 'CPU'],
        'precision': ['fp32', 'int8'],
        'speedup': '1-3x'
    },
    'onnx': {
        'target': ['OpenVINO', 'CoreML', 'DirectML'],
        'precision': ['fp32'],
        'speedup': '1-2x'
    }
}
```

---

## 七、数据集与评估指标

### 7.1 支持的数据集

| 数据集 | 样本数 | 类别数 | 用途 |
|--------|--------|--------|------|
| MVTec AD | 5,354 | 15 | 主测试集 |
| VisA | 10,821 | 12 | 补充验证 |
| BTAD | 2,830 | 3 | 真实工业 |
| MPDD | 1,058 | 6 | 金属表面 |

### 7.2 评估指标

```python
EVALUATION_METRICS = {
    'image_level': {
        'AUROC': 'ROC曲线下面积',
        'AUPRC': 'PR曲线下面积',
        'F1-Max': '最大F1分数'
    },
    'pixel_level': {
        'AUPRO': '区域级AUROC',
        'IoU': '交并比',
        'DICE': 'Dice系数'
    },
    'efficiency': {
        'params': '模型参数量',
        'latency': '推理延迟(ms)',
        'memory': 'GPU内存(MB)'
    }
}
```

---

## 八、实现计划 (云端GPU + 边缘部署)

### Phase 1: 基础框架 (Week 1-2)
- [ ] 项目结构搭建
- [ ] 数据加载模块 (MVTec AD, VisA)
- [ ] 基础异常检测模型实现 (PatchCore, PaDiM基准)
- [ ] 搜索空间定义 (编码器、检测头、损失函数)
- [ ] 奖励函数设计 (边缘部署权重)

### Phase 1: 基础框架 (Week 1-2)
- [ ] 项目结构搭建
- [ ] 数据加载模块 (MVTec AD, VisA)
- [ ] 基础异常检测模型实现 (PatchCore, PaDiM基准)
- [ ] 搜索空间定义 (编码器、检测头、损失函数)
- [ ] **数据增强模块 (AnomalyAugmentor) ⭐**
- [ ] CutPaste/Copy-Paste增强实现

### Phase 2: NAS核心 (Week 3-4)
- [ ] RL控制器实现 (RNN-based)
- [ ] 超网络权重共享机制
- [ ] 多保真度评估器 (成本感知)
- [ ] 性能预测器
- [ ] 成本感知调度器

### Phase 3: LLM+RAG集成与经验学习 (Week 5-7)
- [ ] RAG知识库构建 (论文、架构案例)
- [ ] LLM Agent实现 (MiniMax M2.1集成) ⭐
- [ ] 架构建议生成
- [ ] 搜索结果分析
- [ ] 检索效果评估
- [ ] **经验积累器实现 (ExperienceAccumulator)**
- [ ] **知识蒸馏模块 (KnowledgeDistiller)**
- [ ] **经验感知搜索 (ExperienceAwareSearch)**
- [ ] 定期经验同步与知识更新

### Phase 4: 边缘部署优化 (Week 8-9)
- [ ] TensorRT导出 (NVIDIA Jetson)
- [ ] TFLite导出 (Raspberry Pi)
- [ ] 模型量化 (INT8)
- [ ] 跨设备基准测试
- [ ] 最终模型选择 (Pareto前沿)

### Phase 5: 完善与文档 (Week 9-10)
- [ ] API/CLI接口开发
- [ ] 配置YAML化
- [ ] 单元测试覆盖
- [ ] README文档编写

---

## 九、目录结构

```
InduDet-Search/
├── src/
│   ├── nas/
│   │   ├── controller.py       # RL控制器
│   │   ├── search_space.py     # 搜索空间定义
│   │   ├── supernet.py         # 超网络
│   │   └── evaluator.py        # 评估器
│   │
│   ├── models/
│   │   ├── encoders/          # 编码器
│   │   ├── heads/             # 检测头
│   │   └── anomaly_detectors.py # 异常检测模型
│   │
│   ├── llm/
│   │   ├── agent.py           # LLM Agent
│   │   ├── rag.py             # RAG系统
│   │   ├── prompter.py        # Prompt模板
│   │   └── experience.py      # 经验积累与持续学习 ⭐ 新增
│   │
│   ├── data/
│   │   ├── datasets.py        # 数据集加载
│   │   ├── preprocessing.py   # 预处理
│   │   └── augmentation.py    # 异常生成增强 ⭐
│   │
│   └── utils/
│       ├── metrics.py         # 评估指标
│       ├── trainer.py         # 训练器
│       ├── exporter.py        # 模型导出
│       └── evaluator.py       # 多维度评估器 ⭐
│
├── knowledge/
│   ├── papers/                # 论文PDF
│   ├── embeddings/            # 向量索引
│   ├── cases/                # 架构案例
│   └── experiences/           # 搜索经验积累 ⭐ 新增
│       ├── raw/              # 原始经验记录
│       ├── patterns/         # 分析后的模式
│       └── distilled/        # 蒸馏后的知识
│
├── configs/
│   ├── search_space.yaml      # 搜索空间配置
│   ├── training.yaml          # 训练配置
│   └── llm.yaml               # LLM配置 (MiniMax M2.1) ⭐
│
└── tests/
    ├── test_nas.py
    ├── test_models.py
    └── test_integration.py
```

---

## 十、可行性分析

### 技术可行性: ✅ 高度可行

| 组件 | 成熟度 | 说明 |
|------|--------|------|
| NAS | 高 | DARTS, ENAS等已有成熟实现 |
| RL Controller | 高 | NASNet风格方案广泛验证 |
| 权重共享 | 高 | 超网络技术成熟 |
| 多保真度搜索 | 中高 | 成功应用于多项目 |
| LLM API集成 | 高 | MiniMax M2.1 低延迟低成本 ⭐ |
| RAG系统 | 高 | LangChain+向量库方案成熟 |
| TensorRT导出 | 高 | NVIDIA官方支持 |

### 用户场景评估

| 维度 | 评估 | 建议 |
|------|------|------|
| **LLM选择** | MiniMax M2.1 | 国内访问延迟低，成本低，中文优化 ⭐ |
| **部署目标** | 边缘设备 | 已内置TensorRT/TFLite支持 |
| **GPU资源** | 并行智算云 RTX 5090/4090/3090 | 多卡训练支持 ⭐ |
| **推荐配置** | 4-8卡RTX 4090 | 性价比最优 |
| **TensorRT支持** | ✅ 完美支持 | 5090/4090原生支持 |

### 预期搜索效果

```
Pareto前沿预期 (基于RTX 4090 4卡 + 数据增强):

效率优先架构:
- 参数量: 2-5M
- 延迟: <30ms (TensorRT INT8)
- AUROC: 0.90-0.94 (MVTec AD平均) ⭐ +2% from augmentation

平衡架构:
- 参数量: 8-15M
- 延迟: <50ms (TensorRT FP16)
- AUROC: 0.94-0.97

精度优先架构:
- 参数量: 20-40M
- 延迟: <100ms (TensorRT FP16)
- AUROC: 0.97-0.99

数据增强预期提升:
- 使用CutPaste增强: AUROC +1~3%
- 增强对罕见异常类型的召回: +5~10%
- 模型泛化能力: 跨数据集测试 +2~5%
```

### 多GPU训练支持 (并行智算云)

```python
class MultiGPUTrainingManager:
    """
    多GPU训练管理器 - 针对并行智算云优化
    支持: RTX 5090/4090/3090, 1/2/4/8卡
    """

    def __init__(self, num_gpus: int = 4):
        self.num_gpus = num_gpus
        self.device_ids = list(range(num_gpus))

        # TensorRT优化配置
        self.trt_config = {
            'rtx_4090': {
                'precision': 'fp16',  # 最佳性价比
                'workspace': 4096,     # MB
                'max_batch_size': 32
            },
            'rtx_5090': {
                'precision': 'fp16',  # 支持更好的优化
                'workspace': 8192,
                'max_batch_size': 64
            },
            'rtx_3090': {
                'precision': 'fp16',
                'workspace': 4096,
                'max_batch_size': 32
            }
        }

    def parallelize_model(self, model: nn.Module) -> nn.Module:
        """模型并行化"""
        if self.num_gpus > 1:
            model = nn.DataParallel(model, device_ids=self.device_ids)
        return model

    def get_trt_builder(self, gpu_model: str = 'rtx_4090'):
        """获取TensorRT优化器"""
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                     self.trt_config[gpu_model]['workspace'] * 1024 * 1024)

        return builder, config

### 潜在挑战与解决方案

| 挑战 | 解决方案 |
|------|----------|
| 搜索空间过大 | 1. 先搜索高层架构再微调参数<br>2. 多保真度快速筛选<br>3. 性能预测器预筛选 |
| 多GPU并行训练 | 1. DataParallel/DeepSpeed并行<br>2. 梯度累积适配batch size<br>3. 异步评估减少等待 |
| 计算成本控制 | 1. 成本感知调度器实时监控<br>2. 严格early stopping<br>3. 代金券预算上限保护 |
| LLM API费用 | 1. LLM仅用于分析(非每轮搜索)<br>2. 缓存检索结果<br>3. 使用gpt-3.5-turbo降低开销 |
| 边缘部署精度损失 | 1. 量化感知训练<br>2. TensorRT INT8校准<br>3. 多精度测试 |
| 经验积累效率 | 1. 定期同步到RAG<br>2. LLM蒸馏提炼知识<br>3. 成功/失败模式自动分析 |

---

## 十一、验证计划

### 验证步骤

1. **单元测试**
   - 测试各模块输入输出
   - 测试搜索空间定义正确性
   - 测试RL控制器收敛性
   - 测试多维度评估器计算正确性

2. **集成测试**
   - 端到端搜索流程测试
   - MiniMax M2.1 API集成测试 ⭐
   - LLM+RAG交互测试
   - 多保真度评估正确性

3. **多维度评估验证**
   - 准确度评估: 在MVTec AD上验证AUROC/F1计算
   - 效率评估: 验证参数量、延迟测量准确性
   - 专家相似度验证: 对比PatchCore等架构的相似度分数
   - 消融实验: 各维度权重对最终选择的影响

4. **数据增强验证** ⭐
   - CutPaste增强效果对比 (有无增强的AUROC差异)
   - 不同异常生成方法的对比
   - 异常强度对模型泛化能力的影响
   - 增强样本比例的最优搜索

5. **性能基准**
   - 在MVTec AD上对比搜索到的架构与PatchCore/PaDiM基准
   - 验证搜索效率（vs随机搜索）
   - 评估模型效率和精度权衡 (Pareto前沿)
   - 多维度分数与最终AUROC的相关性分析

6. **消融实验**
   - MiniMax M2.1 vs GPT-4 分析质量对比 ⭐
   - LLM Agent的影响
   - RAG检索效果
   - 专家相似度对搜索方向的引导效果
   - 数据增强对鲁棒性的提升
   - 不同RL策略对比

---

## 十二、参考文献

1. NASNet: "Learning Transferable Architectures for Scalable Image Recognition" (CVPR 2018)
2. DARTS: "Differentiable Architecture Search" (ICLR 2019)
3. PatchCore: "Towards Total Recall in Industrial Anomaly Detection" (CVPR 2022)
4. PaDiM: "Patch Descriptor Distillation" (CVPR 2021)
5. ENAS: "Efficient Neural Architecture Search" (ICLR 2019)
6. NNI: Microsoft Neural Network Intelligence
