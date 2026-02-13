# InduDet-Search: 增量式工业异常检测架构搜索系统

## 一、设计理念（简化版）

参考 SR-LLM 论文核心思想：
- **增量式搜索**：逐步改进架构，而非一次性大规模搜索
- **检索增强**：利用历史成功案例指导新搜索
- **简单优先**：先验证核心假设，再逐步复杂化

### 核心原则
1. 搜索策略：**随机搜索 + 经验引导**（简化RL控制器）
2. 评估策略：**多保真度逐步精化**
3. 知识利用：**RAG检索成功案例**
4. 部署策略：**按需量化导出**

---

## 二、系统架构（简化版）

```
┌─────────────────────────────────────────────────────────────┐
│                  InduDet-Search 简化架构                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐                    │
│  │   LLM Agent  │◄──►│     RAG      │                    │
│  │  (架构建议)    │    │  (案例检索)   │                    │
│  └──────┬───────┘    └──────┬───────┘                    │
│         │                    │                              │
│         ▼                    ▼                              │
│  ┌─────────────────────────────────────────────┐          │
│  │           Search Strategy (搜索策略)            │          │
│  │  ┌─────────────────────────────────────────┐ │          │
│  │  │  Step 1: 随机采样 (探索)                │ │          │
│  │  │  Step 2: 贝叶斯优化 (利用)              │ │          │
│  │  │  Step 3: 经验引导 (RAG检索)             │ │          │
│  │  └─────────────────────────────────────────┘ │          │
│  └───────────────────────────┬─────────────────┘          │
│                              ▼                              │
│  ┌─────────────────────────────────────────────┐          │
│  │           Evaluator (评估器)                    │          │
│  │  ├── Low-Fidelity (1 epoch, 10% data)      │          │
│  │  ├── Medium-Fidelity (10 epochs, 50% data)  │          │
│  │  └── High-Fidelity (50 epochs, 100% data)   │          │
│  └───────────────────────────┬─────────────────┘          │
│                              ▼                              │
│  ┌─────────────────────────────────────────────┐          │
│  │           Experience Manager                   │          │
│  │  ├── 成功案例存储 (successful trials)         │          │
│  │  ├── 失败案例分析 (failure patterns)          │          │
│  │  └── RAG索引更新                            │          │
│  └─────────────────────────────────────────────┘          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、简化设计 vs 原设计对比

| 组件 | 原设计（复杂） | 简化版 |
|------|---------------|--------|
| 搜索控制器 | RL (RNN/LSTM) | 随机搜索 + 贝叶斯优化 |
| 经验利用 | ExperienceAccumulator | 简单案例存储与检索 |
| 评估器 | 多维度(4维) | 准确度 + 效率(2维) |
| 权重共享 | SuperNet | 不需要 |
| 性能预测 | PerformancePredictor | 不需要 |
| 数据增强 | AnomalyAugmentor | 可选 |

**预计实现时间：4-6周**（原计划10周）

---

## 四、核心模块设计

### 4.1 搜索空间（精简版）

```python
# 简化的搜索空间定义
SEARCH_SPACE = {
    # 编码器选择（5选1）
    'backbone': ['ResNet18', 'ResNet50', 'EfficientNet-B0', 'MobileNetV3', 'ViT-Small'],

    # 特征层级（3选1）
    'feature_levels': ['L2', 'L2+L3', 'L2+L3+L4'],

    # 检测方法（3选1）
    'method': ['memory_bank', 'distribution', 'contrastive'],

    # 记忆库大小（3选1）
    'memory_size': [500, 1000, 2000],

    # k-NN参数（3选1）
    'k': [1, 5, 9],
}

# 总搜索空间大小：5 × 3 × 3 × 3 × 3 = 405 种组合
```

### 4.2 搜索策略

```python
from typing import Dict, List, Optional
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class IncrementalSearcher:
    """
    增量式架构搜索器

    搜索策略：
    1. Phase 1: 随机探索 (前50次)
    2. Phase 2: 贝叶斯优化 (基于历史结果)
    3. Phase 3: 经验引导 (RAG检索相似成功案例)
    """

    def __init__(self, search_space: Dict, rag_system, n_random: int = 50):
        self.search_space = search_space
        self.rag = rag_system
        self.n_random = n_random  # 随机探索次数

        # 历史记录
        self.history: List[Dict] = []

        # 贝叶斯优化模型
        self.gp_model = None
        self.X_train = []
        self.y_train = []

    def sample_architecture(self, iteration: int) -> Dict:
        """采样一个架构配置"""
        if iteration < self.n_random:
            # Phase 1: 随机探索
            return self._sample_random()
        elif self.gp_model is None or len(self.history) < self.n_random + 10:
            # Phase 2a: 收集足够数据后再用贝叶斯
            return self._sample_random()
        else:
            # Phase 2b: 贝叶斯优化
            return self._sample_bayesian()

    def _sample_random(self) -> Dict:
        """随机采样"""
        config = {}
        for key, options in self.search_space.items():
            config[key] = np.random.choice(options)
        return config

    def _sample_bayesian(self) -> Dict:
        """贝叶斯优化采样"""
        # 1. 获取候选点
        candidates = self._generate_candidates(n=100)

        # 2. 预测每个候选的期望改善 (EI)
        X_candidates = self._encode_candidates(candidates)
        mu, sigma = self.gp_model.predict(X_candidates, return_std=True)

        # 3. 计算 Expected Improvement
        best_y = max(self.y_train)
        ei = self._compute_ei(mu, sigma, best_y)

        # 4. 选择EI最大的候选
        best_idx = np.argmax(ei)
        return candidates[best_idx]

    def _compute_ei(self, mu: np.ndarray, sigma: np.ndarray, best_y: float) -> np.ndarray:
        """计算Expected Improvement"""
        with np.errstate(divide='warn'):
            improvement = (mu - best_y)
            Z = improvement / sigma
            from scipy.stats import norm
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei

    def _generate_candidates(self, n: int) -> List[Dict]:
        """生成候选配置"""
        candidates = []
        for _ in range(n):
            config = {}
            for key, options in self.search_space.items():
                config[key] = np.random.choice(options)
            candidates.append(config)
        return candidates

    def _encode_candidates(self, candidates: List[Dict]) -> np.ndarray:
        """将配置编码为向量"""
        encoded = []
        keys = list(self.search_space.keys())

        for config in candidates:
            vector = []
            for key in keys:
                options = self.search_space[key]
                value = config[key]
                # one-hot编码
                vec = [1.0 if v == value else 0.0 for v in options]
                vector.extend(vec)
            encoded.append(vector)

        return np.array(encoded)

    def update(self, config: Dict, score: float):
        """更新模型"""
        self.history.append({'config': config, 'score': score})

        # 更新训练数据
        X = self._encode_candidates([config])[0]
        self.X_train.append(X)
        self.y_train.append(score)

        # 重新训练GP
        if len(self.X_train) >= 10:
            X_train = np.array(self.X_train)
            y_train = np.array(self.y_train)
            self.gp_model = GaussianProcessRegressor(
                kernel=Matern(nu=2.5),
                alpha=1e-6,
                normalize_y=True
            )
            self.gp_model.fit(X_train, y_train)
```

### 4.3 RAG 经验检索（简化版）

```python
class SimpleExperienceRAG:
    """
    简化版RAG经验系统

    功能：
    1. 存储成功/失败案例
    2. 检索相似成功案例
    3. 提供架构改进建议
    """

    def __init__(self):
        # 简单案例存储 (可用JSON文件或SQLite)
        self.successful_cases = []  # {'config': dict, 'score': float, 'context': dict}
        self.failed_cases = []      # {'config': dict, 'score': float, 'reason': str}

    def add_case(self, config: Dict, score: float, context: Dict = None):
        """添加搜索案例"""
        case = {
            'config': config,
            'score': score,
            'context': context or {},
            'timestamp': datetime.now().isoformat()
        }

        if score > 0.85:  # 高分案例
            self.successful_cases.append(case)
        elif score < 0.50:  # 失败案例
            case['reason'] = self._analyze_failure_reason(config, score)
            self.failed_cases.append(case)

        # 保持最近1000条
        if len(self.successful_cases) > 1000:
            self.successful_cases = self.successful_cases[-1000:]

    def retrieve_similar(self, current_config: Dict, top_k: int = 3) -> List[Dict]:
        """检索相似的成功案例"""
        if not self.successful_cases:
            return []

        # 简单匹配：统计相同配置的比例
        best_cases = sorted(
            self.successful_cases,
            key=lambda x: x['score'],
            reverse=True
        )[:top_k]

        return best_cases

    def get_design_suggestions(self, target_metrics: Dict) -> List[str]:
        """基于历史经验生成设计建议"""
        suggestions = []

        if not self.successful_cases:
            return ["建议从ResNet50 + memory_bank开始"]

        # 分析高分案例的模式
        backbone_counts = {}
        method_counts = {}

        for case in self.successful_cases[-100:]:  # 最近100个成功案例
            config = case['config']
            backbone_counts[config.get('backbone', '')] = \
                backbone_counts.get(config.get('backbone', ''), 0) + 1
            method_counts[config.get('method', '')] = \
                method_counts.get(config.get('method', ''), 0) + 1

        # 生成建议
        if backbone_counts:
            best_backbone = max(backbone_counts, key=backbone_counts.get)
            suggestions.append(f"Backbone推荐: {best_backbone} (成功率: {backbone_counts[best_backbone]}次)")

        if method_counts:
            best_method = max(method_counts, key=method_counts.get)
            suggestions.append(f"Method推荐: {best_method} (成功率: {method_counts[best_method]}次)")

        return suggestions

    def _analyze_failure_reason(self, config: Dict, score: float) -> str:
        """分析失败原因"""
        if score < 0.5:
            return "检测性能过低，建议更换backbone"
        elif score < 0.7:
            return "性能一般，建议增加memory_size或调整k值"
        return "未知原因"
```

### 4.4 多保真度评估器

```python
class MultiFidelityEvaluator:
    """
    多保真度评估器

    策略：
    - Low: 1 epoch, 10%数据, ~30秒
    - Medium: 10 epochs, 50%数据, ~5分钟
    - High: 50 epochs, 100%数据, ~20分钟
    """

    def __init__(self, dataset, model_builder):
        self.dataset = dataset
        self.model_builder = model_builder

        # 阈值配置
        self.thresholds = {
            'low': 0.60,   # Low分数低于此值则停止
            'medium': 0.75, # Medium分数低于此值则停止
        }

    def evaluate(self, config: Dict, fidelity: str = 'low') -> Dict:
        """
        评估架构

        Args:
            config: 架构配置
            fidelity: 'low' | 'medium' | 'high'

        Returns:
            {'auroc': float, 'latency_ms': float, 'params': int, 'fidelity': str}
        """
        # 获取数据子集
        data = self._get_subset(fidelity)

        # 构建模型
        model = self.model_builder.build(config)

        # 训练
        if fidelity == 'low':
            epochs = 1
        elif fidelity == 'medium':
            epochs = 10
        else:
            epochs = 50

        model.train(data['train'], epochs=epochs)

        # 评估
        metrics = model.evaluate(data['val'])

        # 效率指标
        metrics['params'] = model.count_parameters()
        metrics['latency_ms'] = model.measure_latency(data['test'])
        metrics['fidelity'] = fidelity

        return metrics

    def _get_subset(self, fidelity: str) -> Dict:
        """获取数据子集"""
        if fidelity == 'low':
            # 10%数据
            return {
                'train': self.dataset.sample(0.1),
                'val': self.dataset.sample(0.1),
                'test': self.dataset.sample(0.1)
            }
        elif fidelity == 'medium':
            # 50%数据
            return {
                'train': self.dataset.sample(0.5),
                'val': self.dataset.sample(0.5),
                'test': self.dataset.sample(0.5)
            }
        else:
            # 100%数据
            return {
                'train': self.dataset,
                'val': self.dataset,
                'test': self.dataset
            }

    def should_continue(self, current_metrics: Dict, fidelity: str) -> bool:
        """判断是否需要更高保真度评估"""
        if fidelity == 'low':
            return current_metrics['auroc'] >= self.thresholds['low']
        elif fidelity == 'medium':
            return current_metrics['auroc'] >= self.thresholds['medium']
        return True
```

### 4.5 LLM Agent（简化版）

```python
class SimpleLLMAgent:
    """
    简化版LLM Agent

    功能：
    1. 基于当前搜索状态生成架构建议
    2. 分析失败原因
    3. 提供设计洞察
    """

    def __init__(self, llm_client, rag_system):
        self.llm = llm_client
        self.rag = rag_system

    def suggest_architecture(self, task_spec: Dict, history_summary: Dict) -> Dict:
        """
        基于任务规格和历史结果生成架构建议

        Args:
            task_spec: {'dataset': 'MVTec', 'target_device': 'jetson_nano', ...}
            history_summary: {'best_score': 0.92, 'common_backbone': 'ResNet50', ...}

        Returns:
            {'config': Dict, 'reasoning': str}
        """
        # 检索相关经验
        experiences = self.rag.retrieve_similar(task_spec)

        # 构建Prompt
        prompt = f"""
        为工业异常检测任务设计架构。

        任务要求:
        - 数据集: {task_spec.get('dataset', 'MVTec AD')}
        - 部署设备: {task_spec.get('target_device', 'GPU')}
        - 延迟约束: {task_spec.get('latency_constraint', '100ms')}

        历史最佳:
        - 最高AUROC: {history_summary.get('best_score', '未知')}
        - 常用Backbone: {history_summary.get('common_backbone', 'ResNet50')}
        - 常用Method: {history_summary.get('common_method', 'memory_bank')}

        成功案例参考:
        {self._format_cases(experiences)}

        请输出JSON格式的建议配置，解释设计理由。
        """

        response = self.llm.generate(prompt)

        return self._parse_response(response)

    def _format_cases(self, cases: List[Dict]) -> str:
        """格式化案例"""
        if not cases:
            return "无成功案例，建议使用默认配置"

        formatted = []
        for i, case in enumerate(cases[:3]):
            formatted.append(f"{i+1}. {case['config']} (AUROC: {case['score']})")
        return '\n'.join(formatted)

    def _parse_response(self, response: Dict) -> Dict:
        """解析LLM响应"""
        # 简化处理：假设LLM返回JSON格式
        try:
            import json
            content = response.get('content', '{}')
            if isinstance(content, str):
                result = json.loads(content)
            else:
                result = content

            return {
                'config': {
                    'backbone': result.get('backbone', 'ResNet50'),
                    'feature_levels': result.get('feature_levels', 'L2+L3'),
                    'method': result.get('method', 'memory_bank'),
                    'memory_size': result.get('memory_size', 1000),
                    'k': result.get('k', 9)
                },
                'reasoning': result.get('reasoning', '基于历史经验和任务要求')
            }
        except:
            return {
                'config': {
                    'backbone': 'ResNet50',
                    'feature_levels': 'L2+L3',
                    'method': 'memory_bank',
                    'memory_size': 1000,
                    'k': 9
                },
                'reasoning': '使用默认配置'
            }
```

---

## 五、主搜索流程

```python
def run_search(
    dataset,
    model_builder,
    max_trials: int = 200,
    time_budget_hours: float = 24,
    save_path: str = './results'
):
    """
    主搜索流程

    流程：
    1. 初始化组件
    2. 循环搜索
    3. 记录结果
    4. 保存检查点
    """
    # 初始化
    searcher = IncrementalSearcher(SEARCH_SPACE, rag_system)
    evaluator = MultiFidelityEvaluator(dataset, model_builder)
    llm_agent = SimpleLLMAgent(llm_client, rag_system)
    experience_rag = SimpleExperienceRAG()

    results = []
    best_config = None
    best_score = 0

    start_time = time.time()

    for trial in range(max_trials):
        # 检查时间预算
        elapsed = (time.time() - start_time) / 3600
        if elapsed >= time_budget_hours:
            print(f"时间预算用完 ({elapsed:.1f}小时)")
            break

        # 1. 采样架构
        if trial == 0:
            # 第一个使用LLM建议
            history_summary = {'best_score': best_score}
            suggestion = llm_agent.suggest_architecture(task_spec, history_summary)
            config = suggestion['config']
        else:
            config = searcher.sample_architecture(trial)

        print(f"\nTrial {trial+1}: {config}")

        # 2. 多保真度评估
        metrics = evaluator.evaluate(config, fidelity='low')

        if evaluator.should_continue(metrics, 'low'):
            metrics = evaluator.evaluate(config, fidelity='medium')

        if evaluator.should_continue(metrics, 'medium'):
            metrics = evaluator.evaluate(config, fidelity='high')

        print(f"  AUROC: {metrics['auroc']:.4f}, Latency: {metrics['latency_ms']:.1f}ms")

        # 3. 记录结果
        results.append({
            'trial': trial + 1,
            'config': config,
            'metrics': metrics,
            'elapsed_hours': elapsed
        })

        # 4. 更新组件
        experience_rag.add_case(config, metrics['auroc'])
        searcher.update(config, metrics['auroc'])

        # 5. 更新最佳
        if metrics['auroc'] > best_score:
            best_score = metrics['auroc']
            best_config = config.copy()
            print(f"  New Best! AUROC: {best_score:.4f}")

        # 6. 定期保存
        if (trial + 1) % 20 == 0:
            save_checkpoint(results, best_config, save_path)

    # 最终保存
    save_checkpoint(results, best_config, save_path)

    return results, best_config, best_score


def save_checkpoint(results, best_config, save_path):
    """保存检查点"""
    import json

    os.makedirs(save_path, exist_ok=True)

    with open(f'{save_path}/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    with open(f'{save_path}/best_config.json', 'w') as f:
        json.dump(best_config, f, indent=2)

    print(f"Checkpoint saved to {save_path}")
```

---

## 六、预期效果

### 搜索效率
| 指标 | 简化版 |
|------|--------|
| 总搜索次数 | 200次 |
| 预估GPU时间 | 20-40小时 |
| 预估成本 | ¥50-150 |
| 预期AUROC | 0.90-0.95 |

### 帕累托最优架构
| 类型 | 参数量 | 延迟 | AUROC |
|------|--------|------|-------|
| 轻量 | 2-5M | <30ms | 0.88-0.92 |
| 平衡 | 5-15M | <60ms | 0.92-0.95 |
| 高精 | 15-30M | <100ms | 0.95-0.97 |

---

## 七、实现计划（简化版，4-6周）

### Week 1-2: 基础框架
- [ ] 项目结构搭建
- [ ] 数据加载 (MVTec AD)
- [ ] 基础模型实现 (PatchCore风格)
- [ ] 简化搜索空间定义

### Week 3-4: 核心搜索
- [ ] 随机搜索 + 贝叶斯优化
- [ ] 多保真度评估器
- [ ] 简单RAG经验系统
- [ ] LLM Agent集成 (MiniMax M2.1)

### Week 5-6: 优化与部署
- [ ] 模型导出 (TensorRT/TFLite)
- [ ] 基准测试
- [ ] 结果分析
- [ ] 文档编写

---

## 八、目录结构（简化版）

```
InduDet-Search/
├── src/
│   ├── search/
│   │   ├── searcher.py       # 增量搜索器
│   │   ├── evaluator.py      # 多保真度评估
│   │   └── experience.py     # RAG经验系统
│   │
│   ├── models/
│   │   ├── backbone.py      # 编码器
│   │   ├── head.py          # 检测头
│   │   └── anomaly_detector.py # 完整模型
│   │
│   ├── llm/
│   │   └── agent.py         # LLM Agent
│   │
│   ├── data/
│   │   └── datasets.py     # 数据集
│   │
│   └── utils/
│       ├── metrics.py       # 评估指标
│       └── exporter.py       # 模型导出
│
├── configs/
│   └── config.yaml         # 主配置
│
├── results/                 # 搜索结果
│
└── run_search.py            # 主入口
```

---

## 九、后续迭代方向（可选）

1. **增加RL控制器**：验证简化版效果后，可加入RL进一步优化
2. **增强数据增强**：加入CutPaste等方法
3. **专家相似度评估**：多维度评估
4. **超网络权重共享**：加速搜索
5. **在线学习**：持续改进模型

---

## 十、总结

**简化版设计核心优势**：
1. **实现简单**：去掉复杂组件，易于理解和调试
2. **快速验证**：4-6周可完成核心功能
3. **成本可控**：预估¥50-150
4. **可扩展性**：保留接口，便于后续迭代

**适用场景**：
- 快速原型验证
- 资源有限的研究
- 教学和学习目的
