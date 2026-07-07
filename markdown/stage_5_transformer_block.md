# Stage 5：Transformer Block

## 状态

已完成。

本项目已经完成到 Stage 6：完整 Decoder-only GPT。项目目的旨在学习实现完整 Decoder-only GPT 主干，包括多层 Transformer Block 堆叠、final LayerNorm、`lm_head`、训练、生成、attention 可视化和模型参数统计。接下来后续学习可转至 minimind。

## 学习目标

Stage 5 在 Stage 4 Multi-Head Causal Self-Attention 的基础上，实现一个最小 Transformer Block。

模型流程：

```text
token id
  ↓
token embedding + position embedding
  ↓
Transformer Block
  ├── LayerNorm
  ├── Multi-Head Causal Self-Attention
  ├── Residual Connection
  ├── LayerNorm
  ├── FeedForward
  └── Residual Connection
  ↓
lm_head
  ↓
next-token logits
```

重点理解：

1. Transformer Block 是 GPT 中反复堆叠的基本单元。
2. LayerNorm 用来稳定子层输入。
3. Residual Connection 让输入可以绕过子层直接加到输出上。
4. FeedForward 对每个位置的 hidden state 做非线性变换。
5. block 输入和输出 shape 都是 `[batch_size, sequence_length, n_embd]`。
6. Stage 6 才进入完整 Decoder-only GPT。

## 配置文件

Mac 配置：

```text
configs/transformer_block_lm_mac.yaml
```

RTX 4090 配置：

```text
configs/transformer_block_lm_4090.yaml
```

## 训练命令

Mac 快速测试：

```bash
python -m mini_gpt.train_transformer_block_lm \
  --config configs/transformer_block_lm_mac.yaml \
  --max-iters 20
```

Mac MPS 小规模训练：

```bash
python -m mini_gpt.train_transformer_block_lm \
  --config configs/transformer_block_lm_mac.yaml
```

RTX 4090 24GB 较大配置训练：

```bash
python -m mini_gpt.train_transformer_block_lm \
  --config configs/transformer_block_lm_4090.yaml
```

## 推理命令

Mac 配置 checkpoint：

```bash
python -m mini_gpt.generate_transformer_block_lm \
  --ckpt checkpoints/transformer_block_lm_best.pt \
  --prompt "人工智能"
```

4090 配置 checkpoint：

```bash
python -m mini_gpt.generate_transformer_block_lm \
  --ckpt checkpoints/transformer_block_lm_4090_best.pt \
  --prompt "人工智能"
```

## 可视化命令

```bash
python -m mini_gpt.visualize_transformer_block_attention \
  --ckpt checkpoints/transformer_block_lm_best.pt \
  --prompt "人工智能"
```

默认输出目录：

```text
outputs/transformer_block_attention/
```

## 本阶段边界

Stage 5 聚焦单个 Transformer Block，完整 Decoder-only GPT 主干留到 Stage 6 实现。

## 返回

[返回 README](../README.md)
