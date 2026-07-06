# STAGE_TASKS.md

# 当前阶段：Stage 3 Single-Head Causal Self-Attention

当前项目已经完成：

- Stage 1：字符级 Bigram 语言模型
- Stage 2：Embedding 语言模型 + Mac/4090 双设备配置

现在进入 Stage 3：Single-Head Causal Self-Attention。

本文件描述当前阶段的任务边界。后续实现代码时，必须优先遵守本文件。

## 本阶段核心目标

在 Stage 2 `EmbeddingLanguageModel` 的基础上加入单头 causal self-attention，让学生理解 GPT 中 attention 的最小可运行形态。

本阶段要学习：

1. 在 token embedding + position embedding 之后得到 hidden states。
2. 使用一个 attention head 生成 Q、K、V。
3. 计算 attention score：`QK^T / sqrt(d)`。
4. 使用 causal mask 屏蔽未来 token。
5. 使用 softmax 得到 attention weight。
6. 使用 `attention weight @ V` 得到当前位置聚合后的表示。
7. 将 attention 输出交给 `lm_head`，继续做 next-token prediction。
8. 支持 Mac MPS 小规模调试。
9. 支持 RTX 4090 24GB 较大配置训练。

## Stage 2 到 Stage 3 的模型流程

Stage 2：

```text
token id
  ↓
token embedding + position embedding
  ↓
lm_head
  ↓
next-token logits
```

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

关键张量 shape：

```text
idx:              [batch_size, block_size]
token_emb:        [batch_size, block_size, n_embd]
position_emb:     [block_size, n_embd]
x:                [batch_size, block_size, n_embd]
q, k, v:          [batch_size, block_size, head_size]
attention_score:  [batch_size, block_size, block_size]
attention_weight: [batch_size, block_size, block_size]
attention_out:    [batch_size, block_size, head_size]
logits:           [batch_size, block_size, vocab_size]
```

## 本阶段允许修改和新增

Stage 3 代码实现时，可以根据教学清晰度选择在现有 Stage 2 文件上演进，也可以新增 attention 相关文件。

允许修改：

- `mini_gpt/train_embedding_lm.py`
- `mini_gpt/generate_embedding_lm.py`
- `mini_gpt/train_bigram.py`
- `mini_gpt/generate_bigram.py`
- `mini_gpt/utils.py`
- `configs/`
- 项目级文档

允许新增：

- `mini_gpt/attention.py`
- `mini_gpt/train_attention_lm.py`
- `mini_gpt/generate_attention_lm.py`
- `mini_gpt/training.py`
- `configs/single_head_attention_mac.yaml`
- `configs/single_head_attention_4090.yaml`

## 本阶段通用模块抽取

Stage 3 会把前后多个阶段都能复用的逻辑抽成通用模块，但必须保证 Stage 1 和 Stage 2 原有命令继续可用。

通用模块：

- `mini_gpt/training.py`：Stage 3 抽取，提供 `load_config` 和 `estimate_loss`。
- `mini_gpt/utils.py`：Stage 3 修改，继续提供设备选择、checkpoint 保存/加载，并新增 tokenizer 路径查找。

代码注释要求：

- Stage 3 新增文件顶部 docstring 要写明 `Stage 3 新增` 或 `Stage 3 抽取`。
- Stage 3 新增类和函数的 docstring 要写明 `Stage 3 新增`。
- Stage 3 修改旧文件时，要在模块 docstring 或关键逻辑附近写明 `Stage 3 修改`。
- 注释要标清楚新增/修改点，但不要给每一行都加注释。

## 本阶段禁止实现

本阶段只实现 single-head causal self-attention，禁止提前实现：

- Multi-Head Attention
- Transformer Block
- LayerNorm
- FFN
- LoRA
- SFT
- RAG
- `transformers` / `datasets` / `peft` / `accelerate` / `langchain` / `llama-index`

## 本阶段配置要求

后续实现 Stage 3 时，建议准备两个配置：

```text
configs/
├── single_head_attention_mac.yaml
└── single_head_attention_4090.yaml
```

Mac MPS 小规模调试配置建议：

- `block_size`: 64
- `n_embd`: 128
- `head_size`: 128
- `batch_size`: 16 或 32
- `max_iters`: 1000 左右
- `device`: auto

RTX 4090 24GB 较大训练配置建议：

- `block_size`: 256
- `n_embd`: 256
- `head_size`: 256
- `batch_size`: 64 或 128
- `max_iters`: 10000 左右
- `device`: auto

## 本阶段计划验收命令

Mac 快速测试：

```bash
python -m mini_gpt.train_attention_lm \
  --config configs/single_head_attention_mac.yaml \
  --max-iters 20
```

Mac 小规模训练：

```bash
python -m mini_gpt.train_attention_lm \
  --config configs/single_head_attention_mac.yaml
```

RTX 4090 24GB 较大配置训练：

```bash
python -m mini_gpt.train_attention_lm \
  --config configs/single_head_attention_4090.yaml
```

生成：

```bash
python -m mini_gpt.generate_attention_lm \
  --ckpt checkpoints/single_head_attention_best.pt \
  --prompt "人工智能"
```

## 验收标准

Stage 3 完成时应满足：

1. 模型仍然能完成字符级 next-token prediction。
2. 训练脚本能在 Mac MPS 上用小配置跑通。
3. 训练脚本能在 RTX 4090 24GB 上使用较大配置。
4. 生成脚本能从 checkpoint 加载模型并生成文本。
5. 代码注释清楚说明 Q、K、V 和 attention 的 shape。
6. 代码没有实现 Multi-Head Attention、Transformer Block、LayerNorm、FFN、LoRA、SFT 或 RAG。

## 已完成阶段回顾

Stage 1 已完成字符级 Bigram 语言模型，用于理解 token、vocab、logits、cross entropy 和 generation。

Stage 2 已完成 Embedding 语言模型，用于理解 token embedding、position embedding、`lm_head`、next-token prediction，以及 Mac/4090 双设备配置。
