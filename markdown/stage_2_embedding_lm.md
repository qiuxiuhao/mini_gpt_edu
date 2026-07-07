# Stage 2：Embedding 语言模型

## 状态

已完成。

## 学习目标

Stage 2 在 Bigram 的基础上，引入更接近 GPT 的输入结构：

```text
token id
  ↓
token embedding
  ↓
position embedding
  ↓
lm_head
  ↓
next-token logits
```

重点理解：

1. token id 不是向量。
2. embedding table 是可学习的向量查表矩阵。
3. position embedding 用来提供位置信息。
4. logits shape 为什么是 `[B, T, V]`。
5. cross entropy 前为什么要 flatten。

## 训练命令

Mac 快速测试：

```bash
python -m mini_gpt.train_embedding_lm \
  --config configs/embedding_lm_mac.yaml \
  --max-iters 20
```

Mac 小规模训练：

```bash
python -m mini_gpt.train_embedding_lm \
  --config configs/embedding_lm_mac.yaml
```

RTX 4090 24GB 较大配置训练：

```bash
python -m mini_gpt.train_embedding_lm \
  --config configs/embedding_lm_4090.yaml
```

## 推理命令

```bash
python -m mini_gpt.generate_embedding_lm \
  --ckpt checkpoints/embedding_lm_best.pt \
  --prompt "人工智能"
```

## 本阶段边界

Stage 2 不实现 Self-Attention、Transformer、SFT、LoRA 或 RAG。

## 返回

[返回 README](../README.md)
