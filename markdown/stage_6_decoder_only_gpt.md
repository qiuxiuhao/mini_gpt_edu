# Stage 6：完整 Decoder-only GPT

## 状态

当前进行中。

## 学习目标

Stage 6 在 Stage 5 Transformer Block 的基础上，实现完整 Decoder-only GPT 主干模型。

模型流程：

```text
token id
  ↓
token embedding + position embedding
  ↓
Transformer Block 1
  ↓
Transformer Block 2
  ↓
...
  ↓
Transformer Block n_layer
  ↓
final LayerNorm
  ↓
lm_head
  ↓
next-token logits
```

重点理解：

1. Decoder-only GPT 是多个 Transformer Block 的堆叠。
2. 每一层 block 的输入和输出 shape 都保持 `[batch_size, sequence_length, n_embd]`。
3. final LayerNorm 位于所有 block 之后、`lm_head` 之前。
4. `lm_head` 将 hidden state 映射为 `[batch_size, sequence_length, vocab_size]`。
5. attention 可视化需要区分 layer 和 head。
6. 参数量主要来自 embedding、attention、FeedForward、LayerNorm 和 `lm_head`。
7. `temperature` 控制 logits 缩放，数值越低越保守。
8. `top_k` 只保留分数最高的 k 个候选 token。
9. Stage 6 仍然使用字符级 tokenizer。

## 配置文件

Mac 配置：

```text
configs/gpt_mac.yaml
```

RTX 4090 配置：

```text
configs/gpt_4090.yaml
```

## 训练命令

Mac 快速测试：

```bash
python -m mini_gpt.train_gpt \
  --config configs/gpt_mac.yaml \
  --max-iters 20
```

Mac MPS 小规模训练：

```bash
python -m mini_gpt.train_gpt \
  --config configs/gpt_mac.yaml
```

RTX 4090 24GB 较大配置训练：

```bash
python -m mini_gpt.train_gpt \
  --config configs/gpt_4090.yaml
```

## 推理命令

Mac 配置 checkpoint：

```bash
python -m mini_gpt.generate_gpt \
  --ckpt checkpoints/gpt_best.pt \
  --prompt "人工智能" \
  --temperature 1.0 \
  --top-k 50
```

4090 配置 checkpoint：

```bash
python -m mini_gpt.generate_gpt \
  --ckpt checkpoints/gpt_4090_best.pt \
  --prompt "人工智能" \
  --temperature 0.9 \
  --top-k 50
```

## 可视化和统计命令

GPT attention 可视化：

```bash
python -m mini_gpt.visualize_gpt_attention \
  --ckpt checkpoints/gpt_best.pt \
  --prompt "人工智能"
```

模型参数统计：

```bash
python -m mini_gpt.model_summary \
  --config configs/gpt_mac.yaml
```

## 本阶段边界

Stage 6 不实现 SFT、LoRA、RAG、BPE tokenizer、KV Cache、Flash Attention、复杂生成策略优化或模型服务部署。

Stage 7 才进入更完整的生成策略优化。

## 返回

[返回 README](../README.md)
