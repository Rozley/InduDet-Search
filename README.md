# InduDet-Search

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red?style=flat-square" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
</p>

> 基于 LLM + RAG + 增量式搜索的工业异常检测架构自动搜索系统

## 特性

- 🤖 **LLM 引导**: 使用 MiniMax M2.1 生成架构建议
- 🔍 **增量式搜索**: 随机探索 + 贝叶斯优化
- 📊 **多保真度评估**: Low → Medium → High 三级评估
- 💾 **经验积累**: RAG 风格的历史经验系统
- 🚀 **边缘部署**: 支持 TensorRT/TFLite 导出

## 安装

```bash
# 克隆项目
git clone https://github.com/yourusername/InduDet-Search.git
cd InduDet-Search

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

```bash
# 运行架构搜索
python run_search.py --n-trials 200 --category bottle

# 或使用配置
python run_search.py --config configs/config.yaml
```

## 项目结构

```
InduDet-Search/
├── configs/          # 配置文件
├── data/            # 数据集目录
├── results/         # 搜索结果
├── src/
│   ├── search/     # 搜索模块
│   ├── models/     # 模型定义
│   ├── llm/       # LLM Agent
│   ├── data/      # 数据加载
│   └── utils/     # 工具函数
├── run_search.py   # 主入口
└── requirements.txt
```

## 使用方法

### 命令行参数

```bash
python run_search.py \
  --data-dir ./data/mvtec \
  --category bottle \
  --n-trials 200 \
  --save-dir ./results \
  --device cuda
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-dir` | `./data/mvtec` | 数据集目录 |
| `--category` | `bottle` | 物体类别 |
| `--n-trials` | `200` | 搜索次数 |
| `--save-dir` | `./results` | 结果保存目录 |
| `--device` | `cuda` | 计算设备 |

## 数据集

系统使用 [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) 数据集。

支持的类别：
- bottle, cable, capsule, carpet, grid, guitar, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper

## 预期效果

| 指标 | 值 |
|------|-----|
| 搜索空间 | 405 种配置 |
| 预期 AUROC | 0.90-0.95 |
| GPU 时间 | 20-40 小时 |

## 部署

详见 [DEPLOY.md](DEPLOY.md)

```bash
# Docker 部署
docker compose up indudet-gpu -d
```

## 许可证

MIT License
