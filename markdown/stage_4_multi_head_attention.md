# Stage 4：Multi-Head Causal Self-Attention

## 状态

已完成。

## 学习目标

Stage 4 在 Stage 3 single-head causal self-attention 的基础上，扩展为多个 causal attention head 并行计算。

模型流程：

```text
token id
  ↓
token embedding + position embedding
  ↓
multi-head causal self-attention
  ↓
concat heads
  ↓
output projection
  ↓
lm_head
  ↓
next-token logits
```

重点理解：

1. 每个 head 都有自己的 Q、K、V。
2. 每个 head 都独立计算 `QK^T / sqrt(head_size)`。
3. 每个 head 都使用 causal mask。
4. 多个 head 的输出会 concat。
5. output projection 把 concat 输出映射回 `n_embd`。
6. 每个 head 都可以单独查看 attention weight。
7. Stage 4 不实现 Transformer Block，Stage 5 才进入 Transformer Block。

## 配置文件

Mac 配置：

```text
configs/multi_head_lm_mac.yaml
```

RTX 4090 配置：

```text
configs/multi_head_lm_4090.yaml
```

## 训练命令

Mac 快速测试：

```bash
python -m mini_gpt.train_multi_head_lm \
  --config configs/multi_head_lm_mac.yaml \
  --max-iters 20
```

Mac MPS 小规模训练：

```bash
python -m mini_gpt.train_multi_head_lm \
  --config configs/multi_head_lm_mac.yaml
```

RTX 4090 24GB 较大配置训练：

```bash
python -m mini_gpt.train_multi_head_lm \
  --config configs/multi_head_lm_4090.yaml
```

## 推理命令

Mac 配置 checkpoint：

```bash
python -m mini_gpt.generate_multi_head_lm \
  --ckpt checkpoints/multi_head_lm_best.pt \
  --prompt "人工智能"
```

4090 配置 checkpoint：

```bash
python -m mini_gpt.generate_multi_head_lm \
  --ckpt checkpoints/multi_head_lm_4090_best.pt \
  --prompt "人工智能"
```

## 可视化命令

```bash
python -m mini_gpt.visualize_multi_head_attention \
  --ckpt checkpoints/multi_head_lm_best.pt \
  --prompt "人工智能"
```

默认输出到：

```text
outputs/attention/
```

## 本阶段边界

Stage 4 不实现 Transformer Block、LayerNorm、FeedForward、Residual Connection、SFT、LoRA 或 RAG。

## 返回

[返回 README](../README.md)
