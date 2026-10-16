# CIFAR-10 Image Classification

> 基于公开 `torchvision ResNet-18` 的可复现 CIFAR-10 图像分类项目。

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9-EE4C2C?logo=pytorch&logoColor=white)
![Accuracy](https://img.shields.io/badge/Test_Accuracy-93.84%25-2ea44f)

## 项目简介

本项目使用 torchvision 提供的公开 ResNet-18 与 ImageNet 预训练权重完成 CIFAR-10
十分类任务。针对 32×32 图像，将首层卷积调整为 3×3、步长 1，并移除初始最大池化层。
训练采用随机裁剪、水平翻转、标签平滑、余弦退火和自动混合精度。

## 实验结果

| 项目 | 结果 |
| --- | ---: |
| 模型 | ResNet-18 |
| 数据集 | CIFAR-10 |
| 训练轮数 | 20 |
| 最佳轮次 | 16 |
| 最佳测试准确率 | **93.84%** |
| GPU | NVIDIA RTX 3090 24GB |
| 训练耗时 | 244.04 秒 |

## 项目结构

```text
Classification_test/
├── configs/                 # YAML 实验配置
├── data/                    # 数据集（Git 忽略）
├── datasets/                # 数据加载与增强
├── docs/                    # 实验结果文档
├── evaluation/              # 评估指标
├── experiments/             # 指标与模型输出
├── losses/                  # 损失函数
├── models/                  # 模型定义
├── scripts/                 # 训练入口
├── tools/                   # 配置、随机种子等工具
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 完整训练

```bash
python scripts/train.py --config configs/resnet18.yaml
```

最佳检查点与逐轮指标会写入：

```text
experiments/resnet18_cifar10/
├── best.pt
└── metrics.json
```

### 3. 快速测试

```bash
python scripts/train.py --config configs/resnet18.yaml --epochs 1
```

## 配置

主要超参数位于 `configs/resnet18.yaml`，包括批大小、学习率、训练轮数、权重衰减、
标签平滑、数据路径与输出路径。修改配置即可复现实验变体，无需改动训练代码。

## 模型与数据来源

- 模型：[torchvision ResNet-18](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- 数据集：[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

详细实验记录见 [`docs/RESULTS.md`](docs/RESULTS.md)。数据集与模型权重不提交到 Git，
以保持仓库轻量整洁。

