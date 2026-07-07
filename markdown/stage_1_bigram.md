# Stage 1：字符级 Bigram 语言模型

## 状态

已完成。

## 学习目标

Stage 1 用最小代码理解语言模型训练流程：

1. 字符如何变成 token id。
2. 如何构造 next-token prediction 数据。
3. logits 是什么。
4. cross entropy loss 如何衡量预测错误。
5. 如何通过训练降低 loss。
6. 如何从 prompt 开始逐字符生成文本。

## 训练命令

```bash
python -m mini_gpt.train_bigram --config configs/bigram.yaml
```

## 推理命令

```bash
python -m mini_gpt.generate_bigram \
  --ckpt checkpoints/bigram_best.pt \
  --prompt "人工智能"
```

## 本阶段边界

Stage 1 只实现字符级 Bigram 语言模型，不实现 Embedding、Self-Attention、Transformer、SFT、LoRA 或 RAG。

## 返回

[返回 README](../README.md)
