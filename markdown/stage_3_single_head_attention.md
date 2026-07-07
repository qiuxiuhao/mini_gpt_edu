# Stage 3：Single-Head Causal Self-Attention

## 状态

已完成。

## 学习目标

Stage 3 在 Stage 2 `EmbeddingLanguageModel` 的基础上加入单头 causal self-attention。

模型流程：

```text
token id
  ↓
token embedding + position embedding
  ↓
single-head causal self-attention
  ↓
lm_head
  ↓
next-token logits
```

重点理解：

1. Q、K、V 都来自同一个 hidden state `x`，但通过不同线性层得到。
2. `QK^T / sqrt(d)` 表示缩放后的注意力分数。
3. causal mask 让当前位置只能看见自己和之前的 token。
4. softmax 把被允许的位置转成概率权重。
5. `attention weight @ V` 得到融合上下文后的表示。
6. 本阶段只有一个 attention head，Multi-Head Attention 留到 Stage 4。

## 配置文件

Mac 配置：

```text
configs/single_head_attention_mac.yaml
```

RTX 4090 配置：

```text
configs/single_head_attention_4090.yaml
```

## 训练命令

Mac 快速测试：

```bash
python -m mini_gpt.train_attention_lm \
  --config configs/single_head_attention_mac.yaml \
  --max-iters 20
```

Mac MPS 小规模训练：

```bash
python -m mini_gpt.train_attention_lm \
  --config configs/single_head_attention_mac.yaml
```

RTX 4090 24GB 较大配置训练：

```bash
python -m mini_gpt.train_attention_lm \
  --config configs/single_head_attention_4090.yaml
```

## 推理命令

Mac 配置 checkpoint：

```bash
python -m mini_gpt.generate_attention_lm \
  --ckpt checkpoints/single_head_attention_best.pt \
  --prompt "人工智能"
```

4090 配置 checkpoint：

```bash
python -m mini_gpt.generate_attention_lm \
  --ckpt checkpoints/single_head_attention_4090_best.pt \
  --prompt "人工智能"
```

## Attention 权重接口

`AttentionLanguageModel.forward(..., return_attention=True)` 会额外返回 attention weight：

```text
attention_weight shape: [batch_size, sequence_length, sequence_length]
```

## 本阶段边界

Stage 3 不实现 Multi-Head Attention、Transformer Block、LayerNorm、FFN、SFT、LoRA 或 RAG。

## 返回

[返回 README](../README.md)
