# Stage 6 完整 Decoder-only GPT 完成说明

当前项目已经完成到 Stage 6：完整 Decoder-only GPT。

本项目的目的旨在学习实现完整 Decoder-only GPT 主干，包括多层 Transformer Block 堆叠、final LayerNorm、`lm_head`、训练、生成、attention 可视化和模型参数统计。

接下来后续学习可转至 minimind。

## 已完成阶段

- Stage 1：字符级 Bigram 语言模型。
- Stage 2：Embedding 语言模型。
- Stage 3：Single-Head Causal Self-Attention。
- Stage 4：Multi-Head Causal Self-Attention。
- Stage 5：Transformer Block。
- Stage 6：完整 Decoder-only GPT。

## Stage 6 核心成果

Stage 6 在 Stage 5 Transformer Block 的基础上，实现完整 Decoder-only GPT 主干模型，让学生理解多个 block 如何堆叠成 GPT。

本阶段学习内容：

1. GPT 主干由 token embedding、position embedding、多个 Transformer Block、final LayerNorm 和 `lm_head` 组成。
2. 多层 Transformer Block 会逐层加工 hidden state。
3. 每一层输入和输出 shape 都保持 `[batch_size, block_size, n_embd]`。
4. final LayerNorm 在所有 block 之后稳定最终 hidden state。
5. `lm_head` 把最终 hidden state 映射到 vocab logits。
6. 继续使用字符级 next-token prediction。
7. 支持 GPT attention 权重可视化。
8. 支持模型参数统计，理解模型规模来自哪些模块。
9. 支持 Mac MPS 小规模调试。
10. 支持 RTX 4090 24GB 较大配置训练。
11. 支持最小 `temperature` 和 `top_k` 生成参数。

## Stage 5 到 Stage 6 的模型流程

Stage 5：

```text
token id
  ↓
token embedding + position embedding
  ↓
单个 Transformer Block
  ↓
lm_head
  ↓
next-token logits
```

Stage 6：

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

## GPT 主干内部流程

教学版 decoder-only 结构：

```text
idx
  ↓
token_embedding(idx) + position_embedding(position_ids)
  ↓
for block in blocks:
    x = block(x)
  ↓
x = final_layer_norm(x)
  ↓
logits = lm_head(x)
```

每个 block 复用 Stage 5 的 Pre-LN Transformer Block。

## 关键张量 Shape

假设：

```text
batch_size = B
sequence_length = T
vocab_size = V
n_embd = C
n_head = H
n_layer = L
head_size = C / H
```

关键 shape：

```text
idx:                    [B, T]
token_emb:              [B, T, C]
position_emb:           [T, C]
x:                      [B, T, C]

block_i input:          [B, T, C]
block_i output:         [B, T, C]
final_ln_out:           [B, T, C]
logits:                 [B, T, V]

logits_flat:            [B * T, V]
targets_flat:           [B * T]
```

Attention 可视化时，每层每个 head 的权重 shape：

```text
single layer attention: [B, H, T, T]
all layer attention:    L 个 [B, H, T, T]
```

## Stage 6 文件

完整 GPT 相关文件：

- `mini_gpt/gpt.py`
- `mini_gpt/train_gpt.py`
- `mini_gpt/generate_gpt.py`
- `mini_gpt/visualize_gpt_attention.py`
- `mini_gpt/model_summary.py`
- `configs/gpt_mac.yaml`
- `configs/gpt_4090.yaml`

复用模块：

- `mini_gpt/tokenizer.py`
- `mini_gpt/dataset.py`
- `mini_gpt/utils.py`
- `mini_gpt/training.py`
- `mini_gpt/transformer_block.py`

## Stage 6 配置

Stage 6 包含两个配置：

```text
configs/
├── gpt_mac.yaml
└── gpt_4090.yaml
```

Mac MPS 小规模调试配置：

- `block_size`: 64
- `n_embd`: 128
- `n_head`: 4
- `n_layer`: 2
- `dropout`: 0.1
- `batch_size`: 32
- `max_iters`: 1000
- `weight_decay`: 0.01
- `device`: auto

RTX 4090 24GB 较大训练配置：

- `block_size`: 256
- `n_embd`: 256
- `n_head`: 8
- `n_layer`: 4
- `dropout`: 0.1
- `batch_size`: 64
- `max_iters`: 20000
- `learning_rate`: 0.0005
- `weight_decay`: 0.01
- `device`: auto

## Stage 6 验收命令

Mac 快速测试：

```bash
python -m mini_gpt.train_gpt \
  --config configs/gpt_mac.yaml \
  --max-iters 20
```

Mac 小规模训练：

```bash
python -m mini_gpt.train_gpt \
  --config configs/gpt_mac.yaml
```

RTX 4090 24GB 较大配置训练：

```bash
python -m mini_gpt.train_gpt \
  --config configs/gpt_4090.yaml
```

生成：

```bash
python -m mini_gpt.generate_gpt \
  --ckpt checkpoints/gpt_best.pt \
  --prompt "人工智能" \
  --temperature 1.0 \
  --top-k 50
```

注意力可视化：

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

## 验收标准

Stage 6 完成时满足：

1. 模型能完成字符级 next-token prediction。
2. GPT 主干包含 token embedding、position embedding、多层 Transformer Block、final LayerNorm 和 `lm_head`。
3. 每个 Transformer Block 复用 Stage 5 的核心结构。
4. 每一层输入和输出 shape 都是 `[B, T, n_embd]`。
5. 训练脚本能在 Mac MPS 上用小配置跑通。
6. 训练脚本能在 RTX 4090 24GB 上使用较大配置。
7. 生成脚本能从 checkpoint 加载模型并生成文本。
8. 可视化接口能查看不同 layer、不同 head 的 attention weight。
9. 模型参数统计能展示总参数量和主要模块参数量。
10. 生成支持最小 `temperature` 和 `top_k` 参数。
