# STAGE_TASKS.md

# 当前阶段：Stage 5 Transformer Block

当前项目已经完成：

- Stage 1：字符级 Bigram 语言模型
- Stage 2：Embedding 语言模型
- Stage 3：Single-Head Causal Self-Attention
- Stage 4：Multi-Head Causal Self-Attention

现在进入 Stage 5：Transformer Block。

本文件描述当前阶段的任务边界。后续实现代码时，必须优先遵守本文件。

## 本阶段核心目标

在 Stage 4 Multi-Head Causal Self-Attention 的基础上，实现最小 Transformer Block，让学生理解 GPT block 的核心结构。

本阶段要学习：

1. Transformer Block 是 GPT 的基本重复单元。
2. 一个 block 内部包含 Multi-Head Causal Self-Attention。
3. 一个 block 内部包含 FeedForward。
4. LayerNorm 用于稳定每个子层的输入。
5. Residual Connection 让输入可以绕过子层直接加到输出上。
6. block 输出仍然保持 `[batch_size, block_size, n_embd]`。
7. block 后继续交给 `lm_head` 做 next-token prediction。
8. 支持 Mac MPS 小规模调试。
9. 支持 RTX 4090 24GB 较大配置训练。

## Stage 4 到 Stage 5 的模型流程

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

Stage 5：

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

## Transformer Block 内部流程

建议使用 pre-norm 教学结构：

```text
x
  ↓
x = x + multi_head_attention(layer_norm_1(x))
  ↓
x = x + feed_forward(layer_norm_2(x))
  ↓
block_out
```

Stage 5 只实现一个 Transformer Block，不堆叠多个 block。

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
idx:                  [B, T]
token_emb:            [B, T, C]
position_emb:         [T, C]
x:                    [B, T, C]

ln1_out:              [B, T, C]
attention_out:        [B, T, C]
after_attention_res:  [B, T, C]

ln2_out:              [B, T, C]
feed_forward_out:     [B, T, C]
block_out:            [B, T, C]

logits:               [B, T, V]
```

Multi-head attention 内部仍复用 Stage 4 的 shape：

```text
single head q/k/v:    [B, T, head_size]
single head score:    [B, T, T]
single head weight:   [B, T, T]
concat output:        [B, T, C]
projected output:     [B, T, C]
```

## 本阶段实现内容

Stage 5 实现 Transformer Block 相关文件，并复用 Stage 4 的 Multi-Head Causal Self-Attention。

新增文件：

- `mini_gpt/transformer_block.py`
- `mini_gpt/transformer_block_lm.py`
- `mini_gpt/train_transformer_block_lm.py`
- `mini_gpt/generate_transformer_block_lm.py`
- `mini_gpt/visualize_transformer_block_attention.py`
- `configs/transformer_block_lm_mac.yaml`
- `configs/transformer_block_lm_4090.yaml`

修改文件：

- `mini_gpt/utils.py`：checkpoint 额外记录 `n_layer`，保持旧阶段兼容。
- `README.md`：同步 Stage 5 目标、配置和命令。

无论新增还是修改，都必须保持 Stage 1、Stage 2、Stage 3、Stage 4 原有命令继续可用。

## 本阶段禁止实现

Stage 5 只实现一个 Transformer Block，禁止提前实现：

- 完整 Decoder-only GPT
- 多层 Transformer Block 堆叠
- SFT
- LoRA
- RAG
- `transformers` / `datasets` / `peft` / `accelerate` / `langchain` / `llama-index`

## 本阶段配置要求

Stage 5 准备两个配置：

```text
configs/
├── transformer_block_lm_mac.yaml
└── transformer_block_lm_4090.yaml
```

Mac MPS 小规模调试配置建议：

- `block_size`: 64
- `n_embd`: 128
- `n_head`: 4
- `n_layer`: 1
- `dropout`: 0.0
- `batch_size`: 16 或 32
- `max_iters`: 1000 左右
- `device`: auto

RTX 4090 24GB 较大训练配置建议：

- `block_size`: 256
- `n_embd`: 256
- `n_head`: 8
- `n_layer`: 1
- `dropout`: 0.1
- `batch_size`: 64 或 128
- `max_iters`: 10000 左右
- `device`: auto

## 本阶段计划验收命令

Mac 快速测试：

```bash
python -m mini_gpt.train_transformer_block_lm \
  --config configs/transformer_block_lm_mac.yaml \
  --max-iters 20
```

Mac 小规模训练：

```bash
python -m mini_gpt.train_transformer_block_lm \
  --config configs/transformer_block_lm_mac.yaml
```

RTX 4090 24GB 较大配置训练：

```bash
python -m mini_gpt.train_transformer_block_lm \
  --config configs/transformer_block_lm_4090.yaml
```

生成：

```bash
python -m mini_gpt.generate_transformer_block_lm \
  --ckpt checkpoints/transformer_block_lm_best.pt \
  --prompt "人工智能"
```

注意力可视化：

```bash
python -m mini_gpt.visualize_transformer_block_attention \
  --ckpt checkpoints/transformer_block_lm_best.pt \
  --prompt "人工智能"
```

## 验收标准

Stage 5 完成时应满足：

1. 模型仍然能完成字符级 next-token prediction。
2. Transformer Block 复用 Stage 4 的 Multi-Head Causal Self-Attention。
3. Transformer Block 包含 LayerNorm、Residual Connection、FeedForward。
4. block 输入和输出 shape 都是 `[B, T, n_embd]`。
5. 训练脚本能在 Mac MPS 上用小配置跑通。
6. 训练脚本能在 RTX 4090 24GB 上使用较大配置。
7. 生成脚本能从 checkpoint 加载模型并生成文本。
8. 可视化接口能查看 block 内 multi-head attention weight。
9. 代码没有实现完整 Decoder-only GPT、SFT、LoRA 或 RAG。

## Stage 5 和 Stage 6 的衔接

Stage 5 只解决一个 Transformer Block。

Stage 6 才进入完整 Decoder-only GPT，届时再引入：

- 多层 Transformer Block 堆叠
- 最终模型结构整理
- 完整 GPT 风格 decoder-only 模型

Stage 5 不要提前实现 Stage 6 的内容。

## 已完成阶段回顾

Stage 1 已完成字符级 Bigram 语言模型，用于理解 token、vocab、logits、cross entropy 和 generation。

Stage 2 已完成 Embedding 语言模型，用于理解 token embedding、position embedding、`lm_head`、next-token prediction，以及 Mac/4090 双设备配置。

Stage 3 已完成 Single-Head Causal Self-Attention，用于理解单个 attention head 中的 Q、K、V、causal mask、softmax attention weight 和 `attention weight @ V`。

Stage 4 已完成 Multi-Head Causal Self-Attention，用于理解多个 attention head 并行、concat 和 output projection。
