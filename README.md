# MiniGPT-edu

本项目用于从零实现一个教学版 GPT 语言模型，目标是帮助零基础学习者从项目实战出发理解大模型核心算法。

本项目会分阶段实现：

1. 字符级 Bigram 语言模型
2. Embedding
3. Self-Attention
4. Multi-Head Attention
5. Transformer Block
6. Decoder-only GPT
7. 生成策略
8. SFT
9. LoRA
10. RAG

## 当前阶段

Stage 1：字符级 Bigram 语言模型。

## 环境配置

```bash
conda create -n mini_gpt_edu python=3.11 -y
conda activate mini_gpt_edu
pip install -r requirements.txt
```

## Stage 1 学习目标

当前阶段只实现字符级 Bigram 语言模型，用最小代码理解语言模型训练流程：

1. 字符如何变成 token id
2. 如何构造 next-token prediction 数据
3. logits 是什么
4. cross entropy loss 如何衡量预测错误
5. 如何通过训练降低 loss
6. 如何从 prompt 开始逐字符生成文本

本阶段不实现 Transformer、Self-Attention、Multi-Head Attention、LoRA 或 RAG。

## 数据准备

默认训练数据已经放在：

```text
data/raw.txt
```

这是一个很小的中文教学文本，适合快速跑通完整流程。

## 训练命令

```bash
python -m mini_gpt.train_bigram --config configs/bigram.yaml
```

训练完成后会保存：

```text
checkpoints/bigram_best.pt
checkpoints/tokenizer.json
```

## 快速测试命令

如果只想确认代码能跑通，可以覆盖训练步数：

```bash
python -m mini_gpt.train_bigram --config configs/bigram.yaml --max-iters 20
```

## 生成命令

```bash
python -m mini_gpt.generate_bigram \
  --ckpt checkpoints/bigram_best.pt \
  --prompt "人工智能"
```
