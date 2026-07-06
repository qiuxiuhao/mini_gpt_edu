# STAGE_TASKS.md

# 当前阶段：Stage 4 Multi-Head Causal Self-Attention

当前项目已经完成：

- Stage 1：字符级 Bigram 语言模型
- Stage 2：Embedding 语言模型
- Stage 3：Single-Head Causal Self-Attention

现在进入 Stage 4：Multi-Head Causal Self-Attention。

本文件描述当前阶段的任务边界。后续实现代码时，必须优先遵守本文件。

## 本阶段核心目标

在 Stage 3 single-head causal self-attention 的基础上，实现 Multi-Head Causal Self-Attention，让学生理解 GPT 中多个 attention head 并行工作的最小形态。

本阶段要学习：

1. 为什么一个 attention head 只能从一个角度学习 token 关系。
2. 为什么需要多个 attention head 并行学习不同关系。
3. 每个 head 都有自己的 Q、K、V。
4. 每个 head 都计算 `QK^T / sqrt(head_size)`。
5. 每个 head 都使用 causal mask，不能看未来 token。
6. 每个 head 都经过 softmax 得到 attention weight。
7. 每个 head 都计算 `attention weight @ V`。
8. 多个 head 的输出在 embedding 维度 concat。
9. concat 后继续交给 `lm_head` 做 next-token prediction。
10. 支持 Mac MPS 小规模调试。
11. 支持 RTX 4090 24GB 较大配置训练。

## Stage 3 到 Stage 4 的模型流程

Stage 3：

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

Stage 4：

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

## 关键张量 Shape

假设：

```text
batch_size = B
sequence_length = T
vocab_size = V
n_embd = C
n_head = H
head_size = C / H
```

关键 shape：

```text
idx:                 [B, T]
token_emb:           [B, T, C]
position_emb:        [T, C]
x:                   [B, T, C]

single head q:       [B, T, head_size]
single head k:       [B, T, head_size]
single head v:       [B, T, head_size]
single head score:   [B, T, T]
causal mask:         [T, T]
single head weight:  [B, T, T]
single head out:     [B, T, head_size]

all head outputs:    H 个 [B, T, head_size]
concat output:       [B, T, C]
projected output:    [B, T, C]
logits:              [B, T, V]
```

Stage 4 要求 `n_embd % n_head == 0`。

## 本阶段允许后续实现的内容

后续进入 Stage 4 代码实现时，可以根据教学清晰度新增或修改 multi-head attention 相关文件。

可以考虑新增：

- `mini_gpt/multi_head_lm.py`
- `mini_gpt/train_multi_head_lm.py`
- `mini_gpt/generate_multi_head_lm.py`
- `mini_gpt/visualize_multi_head_attention.py`
- `configs/multi_head_lm_mac.yaml`
- `configs/multi_head_lm_4090.yaml`

可以考虑修改：

- `mini_gpt/attention.py`
- `mini_gpt/train_attention_lm.py`
- `mini_gpt/generate_attention_lm.py`
- `mini_gpt/training.py`
- `mini_gpt/utils.py`

无论新增还是修改，都必须保持 Stage 1、Stage 2、Stage 3 原有命令继续可用。

## 本阶段禁止实现

Stage 4 只实现 Multi-Head Causal Self-Attention，禁止提前实现：

- Transformer Block
- LayerNorm
- FeedForward
- Residual Connection
- SFT
- LoRA
- RAG
- `transformers` / `datasets` / `peft` / `accelerate` / `langchain` / `llama-index`

## 本阶段配置要求

后续实现 Stage 4 时，建议准备两个配置：

```text
configs/
├── multi_head_lm_mac.yaml
└── multi_head_lm_4090.yaml
```

Mac MPS 小规模调试配置建议：

- `block_size`: 64
- `n_embd`: 128
- `n_head`: 4
- `dropout`: 0.0
- `batch_size`: 16 或 32
- `max_iters`: 1000 左右
- `device`: auto

RTX 4090 24GB 较大训练配置建议：

- `block_size`: 256
- `n_embd`: 256
- `n_head`: 8
- `dropout`: 0.0
- `batch_size`: 64 或 128
- `max_iters`: 10000 左右
- `device`: auto

## 本阶段计划验收命令

Mac 快速测试：

```bash
python -m mini_gpt.train_multi_head_lm \
  --config configs/multi_head_lm_mac.yaml \
  --max-iters 20
```

Mac 小规模训练：

```bash
python -m mini_gpt.train_multi_head_lm \
  --config configs/multi_head_lm_mac.yaml
```

RTX 4090 24GB 较大配置训练：

```bash
python -m mini_gpt.train_multi_head_lm \
  --config configs/multi_head_lm_4090.yaml
```

生成：

```bash
python -m mini_gpt.generate_multi_head_lm \
  --ckpt checkpoints/multi_head_lm_best.pt \
  --prompt "人工智能"
```

注意力可视化：

```bash
python -m mini_gpt.visualize_multi_head_attention \
  --ckpt checkpoints/multi_head_lm_best.pt \
  --prompt "人工智能"
```

## 验收标准

Stage 4 完成时应满足：

1. 模型仍然能完成字符级 next-token prediction。
2. 多个 attention head 并行计算，并且每个 head 都是 causal attention。
3. 每个 head 的 attention weight shape 清楚可解释。
4. 多个 head 的输出 concat 后 shape 为 `[B, T, n_embd]`。
5. output projection 后 shape 仍为 `[B, T, n_embd]`。
6. 训练脚本能在 Mac MPS 上用小配置跑通。
7. 训练脚本能在 RTX 4090 24GB 上使用较大配置。
8. 生成脚本能从 checkpoint 加载模型并生成文本。
9. 可视化接口能查看不同 head 的 attention weight。
10. 代码没有实现 Transformer Block、LayerNorm、FeedForward、Residual Connection、SFT、LoRA 或 RAG。

## Stage 4 和 Stage 5 的衔接

Stage 4 只解决 Multi-Head Causal Self-Attention。

Stage 5 才进入 Transformer Block，届时再引入：

- Residual Connection
- LayerNorm
- FeedForward

Stage 4 不要提前实现 Stage 5 的内容。

## 已完成阶段回顾

Stage 1 已完成字符级 Bigram 语言模型，用于理解 token、vocab、logits、cross entropy 和 generation。

Stage 2 已完成 Embedding 语言模型，用于理解 token embedding、position embedding、`lm_head`、next-token prediction，以及 Mac/4090 双设备配置。

Stage 3 已完成 Single-Head Causal Self-Attention，用于理解单个 attention head 中的 Q、K、V、causal mask、softmax attention weight 和 `attention weight @ V`。
