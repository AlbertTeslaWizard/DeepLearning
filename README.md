# DeepLearning 学习笔记与代码示例

按知识主题整理的深度学习与大语言模型学习示例，包含基础训练、经典网络、Transformer 组件和 MiniLLM 训练与生成。

## 源码目录树

以下展示学习源码、包初始化文件和独立项目配置；省略 Git/工具配置、缓存、数据、模型权重、日志及图片。`__init__.py` 是空的包初始化文件。

```text
DeepLearning/
├── README.md
├── regression/  # 回归入门
│   ├── __init__.py
│   ├── deep_learning_linear_fit.py
│   ├── deep_learning_quadratic_fit_dataloader.py
│   ├── deep_learning_quadratic_fit_tb.py
│   ├── infer_quadratic.py
│   └── train_quadratic_with_validation.py
├── training_techniques/  # 训练技巧
│   ├── __init__.py
│   ├── train_sine_with_early_stopping.py
│   ├── train_sine_with_regularization.py
│   └── train_sine_with_weight_decay_and_scheduler.py
├── evaluation_metrics/  # 评估指标
│   ├── __init__.py
│   ├── demo_sklearn_metrics.py
│   └── sklearn_metrics.py
├── cnn/  # 卷积神经网络
│   ├── __init__.py
│   ├── demo_CNN.py
│   └── demo_CNN_practice.py
├── rnn/  # 循环神经网络
│   ├── __init__.py
│   ├── demo_GRU.py
│   ├── demo_LSTM.py
│   └── demo_RNN.py
├── transformer/  # Transformer
│   ├── blocks/  # TransformerBlock 各版本与多层堆叠
│   │   ├── __init__.py
│   │   ├── demo_TransformerBlock.py
│   │   ├── demo_TransformerBlock_Causal_Mask.py
│   │   ├── demo_TransformerBlock_GQA_SwiGLU_PreRMSNorm.py
│   │   ├── demo_TransformerBlock_MHA.py
│   │   ├── demo_TransformerBlock_PositionalEncoding.py
│   │   ├── demo_TransformerBlock_RoPE.py
│   │   ├── demo_TransformerBlock_practice.py
│   │   └── demo_TransformerStack.py
│   ├── components/  # RoPE、GQA/MQA、RMSNorm、SwiGLU
│   │   ├── __init__.py
│   │   ├── demo_GQA_MQA.py
│   │   ├── demo_RMSNorm.py
│   │   ├── demo_RoPE.py
│   │   └── demo_SwiGLU_FFN.py
│   └── __init__.py
├── llm/  # 大语言模型
│   ├── generation/  # 文本生成与采样策略
│   │   ├── __init__.py
│   │   ├── demo_LLM_CombinedSampling.py
│   │   ├── demo_LLM_Generate.py
│   │   ├── demo_LLM_TemperatureSampling.py
│   │   ├── demo_LLM_TopKSampling.py
│   │   └── demo_LLM_TopPSampling.py
│   ├── modeling/  # 模型骨干与语言模型
│   │   ├── __init__.py
│   │   ├── demo_LLM_Backbone.py
│   │   └── demo_MiniLLM.py
│   ├── tokenization/  # 分词与词嵌入
│   │   ├── __init__.py
│   │   ├── demo_LLM_AttentionMask.py
│   │   ├── demo_LLM_BatchPadding.py
│   │   ├── demo_TokenEmbedding.py
│   │   └── demo_Tokenizer.py
│   ├── training/  # 下一词预测损失与训练
│   │   ├── __init__.py
│   │   ├── demo_LLM_Train.py
│   │   └── demo_NextTokenLoss.py
│   └── __init__.py
├── templates/  # 训练模板
│   ├── __init__.py
│   └── template.py
└── mlp-sine-regression/  # 独立正弦回归项目
    ├── config.yaml
    ├── datasets.py
    ├── evaluate.py
    ├── models.py
    ├── train.py
    └── utils.py
```

## 示例索引

点击文件名即可在 GitHub 上查看源码。

### 回归入门

| 文件 | 学习内容 |
| --- | --- |
| [deep_learning_linear_fit.py](regression/deep_learning_linear_fit.py) | 线性回归与基础训练循环 |
| [deep_learning_quadratic_fit_dataloader.py](regression/deep_learning_quadratic_fit_dataloader.py) | 二次函数拟合、Dataset 与 DataLoader |
| [deep_learning_quadratic_fit_tb.py](regression/deep_learning_quadratic_fit_tb.py) | 二次函数拟合与 TensorBoard |
| [infer_quadratic.py](regression/infer_quadratic.py) | 加载二次函数模型权重进行推理 |
| [train_quadratic_with_validation.py](regression/train_quadratic_with_validation.py) | 训练集与验证集划分、最佳模型保存 |

### 训练技巧

| 文件 | 学习内容 |
| --- | --- |
| [train_sine_with_early_stopping.py](training_techniques/train_sine_with_early_stopping.py) | 正弦拟合与早停 |
| [train_sine_with_regularization.py](training_techniques/train_sine_with_regularization.py) | 正弦拟合与正则化 |
| [train_sine_with_weight_decay_and_scheduler.py](training_techniques/train_sine_with_weight_decay_and_scheduler.py) | 权重衰减与学习率调度 |

### 评估指标

| 文件 | 学习内容 |
| --- | --- |
| [demo_sklearn_metrics.py](evaluation_metrics/demo_sklearn_metrics.py) | 分类报告与混淆矩阵 |
| [sklearn_metrics.py](evaluation_metrics/sklearn_metrics.py) | 分类模型与 accuracy、F1 评估 |

### 卷积神经网络

| 文件 | 学习内容 |
| --- | --- |
| [demo_CNN.py](cnn/demo_CNN.py) | MNIST 卷积分类 |
| [demo_CNN_practice.py](cnn/demo_CNN_practice.py) | CNN 手写练习 |

### 循环神经网络

| 文件 | 学习内容 |
| --- | --- |
| [demo_GRU.py](rnn/demo_GRU.py) | GRU 序列建模 |
| [demo_LSTM.py](rnn/demo_LSTM.py) | LSTM 序列建模 |
| [demo_RNN.py](rnn/demo_RNN.py) | 基础 RNN 序列建模 |

### Transformer

| 文件 | 学习内容 |
| --- | --- |
| [blocks/demo_TransformerBlock.py](transformer/blocks/demo_TransformerBlock.py) | 基于 PyTorch MultiheadAttention 的基础模块 |
| [blocks/demo_TransformerBlock_Causal_Mask.py](transformer/blocks/demo_TransformerBlock_Causal_Mask.py) | 因果注意力模块 |
| [blocks/demo_TransformerBlock_GQA_SwiGLU_PreRMSNorm.py](transformer/blocks/demo_TransformerBlock_GQA_SwiGLU_PreRMSNorm.py) | 组合 GQA、SwiGLU 与前置 RMSNorm |
| [blocks/demo_TransformerBlock_MHA.py](transformer/blocks/demo_TransformerBlock_MHA.py) | 手写多头注意力与 TransformerBlock |
| [blocks/demo_TransformerBlock_PositionalEncoding.py](transformer/blocks/demo_TransformerBlock_PositionalEncoding.py) | 正弦位置编码与 TransformerBlock |
| [blocks/demo_TransformerBlock_RoPE.py](transformer/blocks/demo_TransformerBlock_RoPE.py) | 集成 RoPE 的 TransformerBlock |
| [blocks/demo_TransformerBlock_practice.py](transformer/blocks/demo_TransformerBlock_practice.py) | 基础 TransformerBlock 练习 |
| [blocks/demo_TransformerStack.py](transformer/blocks/demo_TransformerStack.py) | 多层 TransformerBlock 堆叠 |
| [components/demo_GQA_MQA.py](transformer/components/demo_GQA_MQA.py) | MHA、MQA、GQA 对比 |
| [components/demo_RMSNorm.py](transformer/components/demo_RMSNorm.py) | 均方根归一化 |
| [components/demo_RoPE.py](transformer/components/demo_RoPE.py) | 旋转位置编码及数学说明 |
| [components/demo_SwiGLU_FFN.py](transformer/components/demo_SwiGLU_FFN.py) | SwiGLU 门控前馈网络 |

### 大语言模型

| 文件 | 学习内容 |
| --- | --- |
| [generation/demo_LLM_CombinedSampling.py](llm/generation/demo_LLM_CombinedSampling.py) | 组合采样策略 |
| [generation/demo_LLM_Generate.py](llm/generation/demo_LLM_Generate.py) | 自回归文本生成 |
| [generation/demo_LLM_TemperatureSampling.py](llm/generation/demo_LLM_TemperatureSampling.py) | 温度采样 |
| [generation/demo_LLM_TopKSampling.py](llm/generation/demo_LLM_TopKSampling.py) | Top-k 采样 |
| [generation/demo_LLM_TopPSampling.py](llm/generation/demo_LLM_TopPSampling.py) | Top-p（核）采样 |
| [modeling/demo_LLM_Backbone.py](llm/modeling/demo_LLM_Backbone.py) | 词嵌入与 Transformer 骨干 |
| [modeling/demo_MiniLLM.py](llm/modeling/demo_MiniLLM.py) | 添加归一化与语言模型输出头 |
| [tokenization/demo_LLM_AttentionMask.py](llm/tokenization/demo_LLM_AttentionMask.py) | 注意力掩码的形状变换与 padding 分数屏蔽 |
| [tokenization/demo_LLM_BatchPadding.py](llm/tokenization/demo_LLM_BatchPadding.py) | 文本批的 padding 与注意力掩码 |
| [tokenization/demo_TokenEmbedding.py](llm/tokenization/demo_TokenEmbedding.py) | 词嵌入接入 TransformerBlock |
| [tokenization/demo_Tokenizer.py](llm/tokenization/demo_Tokenizer.py) | 文本、token 与 token ID 的转换 |
| [training/demo_LLM_Train.py](llm/training/demo_LLM_Train.py) | 训练 MiniLLM 并保存权重 |
| [training/demo_NextTokenLoss.py](llm/training/demo_NextTokenLoss.py) | 错位标签与下一词预测损失 |

### 训练模板

| 文件 | 学习内容 |
| --- | --- |
| [template.py](templates/template.py) | 基础 PyTorch 训练模板 |

### 独立正弦回归项目

| 文件 | 学习内容 |
| --- | --- |
| [config.yaml](mlp-sine-regression/config.yaml) | 训练超参数与权重保存路径 |
| [datasets.py](mlp-sine-regression/datasets.py) | 正弦回归数据集 |
| [evaluate.py](mlp-sine-regression/evaluate.py) | 加载模型并评估预测 |
| [models.py](mlp-sine-regression/models.py) | 残差 MLP 模型 |
| [train.py](mlp-sine-regression/train.py) | 配置驱动的训练、混合精度与梯度裁剪 |
| [utils.py](mlp-sine-regression/utils.py) | 早停工具 |

## 建议阅读顺序

1. 从 `regression/` 开始，熟悉模型、损失函数、训练循环、数据加载与验证。
2. 阅读 `training_techniques/` 和 `evaluation_metrics/`，学习训练控制与评估。
3. 学习 `cnn/` 与 `rnn/`，对照示例和 `practice` 文件练习。
4. 从基础 TransformerBlock、手写 MHA、位置编码开始，再阅读 `transformer/components/`，最后学习组合模块与 TransformerStack。
5. 阅读 `llm/tokenization/` → `llm/modeling/` → `llm/training/` → `llm/generation/`，串起分词、模型、损失、训练与生成。
6. 用 `mlp-sine-regression/` 复习将数据、模型、配置与训练拆分为独立文件的项目组织方式。

## 运行方式

使用已有的深度学习 Python 环境。代码按需使用 PyTorch、torchvision、transformers、scikit-learn、TensorBoard 与 PyYAML；不同示例需要的依赖不同。

除独立项目外，**在仓库根目录 `DeepLearning/` 使用 `python -m` 运行**，模块路径不带 `.py`。这样包导入及数据、权重等相对路径都能正确解析。

```bash
# 基础回归示例
python -m regression.deep_learning_linear_fit

# 无需分词器或训练权重的 Transformer 示例
python -m transformer.blocks.demo_TransformerBlock_RoPE
python -m transformer.blocks.demo_TransformerStack

# 先训练，再生成；训练会写入或覆盖根目录 mini_llm.pt
python -m llm.training.demo_LLM_Train
python -m llm.generation.demo_LLM_Generate
python -m llm.generation.demo_LLM_TopKSampling
```

其他示例同样将文件路径中的 `/` 换成 `.`，去掉 `.py` 后运行。`demo_RoPE.py` 主要提供组件定义，可通过引用它的 Transformer 示例观察效果。

独立项目仍在自己的目录中运行，以读取本地 `config.yaml`：

```bash
cd mlp-sine-regression
python train.py
python evaluate.py
```

### 数据与权重前提

- CNN 示例使用 MNIST，读取根目录 `data/`；首次运行可能下载数据。
- 使用 `AutoTokenizer` 的示例依赖 GPT-2 分词器，首次运行可能需要联网下载；生成示例需要根目录已有兼容的 `mini_llm.pt`，可由 LLM 训练示例生成。
- 二次函数推理需要先运行 `python -m regression.train_quadratic_with_validation`，生成根目录 `best_model_epoch`。
- 部分正弦训练示例也使用 `best_model_epoch`，会覆盖同名权重。二次函数推理前应确保该文件来自对应的二次函数训练。
- 早停示例使用根目录 `best_model.pth`；独立项目使用其自身目录内配置指定的权重文件。
- 日志与 TensorBoard 输出沿用根目录 `training.log`、`runs/exp_tanh`。已有权重、数据、日志和图片保留原位；它们通常被 Git 忽略，GitHub 克隆不包含这些本地产物。

## 维护约定

新增示例按主要知识主题归类；练习版与对应示例放在一起。新增或移动文件时，同步更新本 README 的目录树、示例索引与相关导入。目录树手动维护，不需要额外工具或 GitHub 插件。
