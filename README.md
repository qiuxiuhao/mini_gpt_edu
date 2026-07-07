# MiniGPT-edu

本项目用于从零实现一个教学版 GPT 语言模型，目标是帮助零基础学习者从项目实战出发理解大模型核心算法。

本项目会分阶段实现：

1. 字符级 Bigram 语言模型
2. Embedding
3. Single-Head Causal Self-Attention
4. Multi-Head Causal Self-Attention
5. Transformer Block
6. Decoder-only GPT
7. 生成策略
8. SFT
9. LoRA
10. RAG

## 当前阶段

Stage 6：完整 Decoder-only GPT。

Stage 1 字符级 Bigram 语言模型已完成。

Stage 2 Embedding 语言模型 + Mac/4090 双设备配置已完成。

Stage 3 Single-Head Causal Self-Attention 已完成。

Stage 4 Multi-Head Causal Self-Attention 已完成。

Stage 5 Transformer Block 已完成。

Stage 6 当前只实现完整 Decoder-only GPT 主干模型，包括多层 Transformer Block 堆叠、final LayerNorm、`lm_head`、训练、生成、attention 可视化和模型参数统计。不实现 SFT、LoRA、RAG、BPE tokenizer、KV Cache 或 Flash Attention。

## 环境配置

```bash
conda create -n mini_gpt_edu python=3.11 -y
conda activate mini_gpt_edu
pip install -r requirements.txt
```

主要依赖保持轻量：

- torch
- numpy
- matplotlib
- pyyaml
- tqdm

当前阶段不使用：

- `transformers`
- `datasets`
- `peft`
- `accelerate`
- `langchain`
- `llama-index`
- BPE tokenizer
- KV Cache
- Flash Attention

## 数据准备

默认小规模训练数据：

```text
data/raw.txt
```

4090 较大配置训练数据：

```text
data/computer_knowledge/raw/full_corpus.txt
```

## Stage 1：字符级 Bigram 语言模型

Stage 1 用最小代码理解语言模型训练流程：

1. 字符如何变成 token id
2. 如何构造 next-token prediction 数据
3. logits 是什么
4. cross entropy loss 如何衡量预测错误
5. 如何通过训练降低 loss
6. 如何从 prompt 开始逐字符生成文本

训练命令：

```bash
python -m mini_gpt.train_bigram --config configs/bigram.yaml
```

生成命令：

```bash
python -m mini_gpt.generate_bigram \
  --ckpt checkpoints/bigram_best.pt \
  --prompt "人工智能"
```

## Stage 2：Embedding 语言模型

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

Stage 2 重点理解 token id 不是向量，embedding table 才是可学习的向量查表矩阵。

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

生成命令：

```bash
python -m mini_gpt.generate_embedding_lm \
  --ckpt checkpoints/embedding_lm_best.pt \
  --prompt "人工智能"
```

## Stage 3：Single-Head Causal Self-Attention

Stage 3 在 Stage 2 `EmbeddingLanguageModel` 的基础上加入单头 causal self-attention。

Stage 3 新增 attention 专属文件：

- `mini_gpt/attention.py`
- `mini_gpt/train_attention_lm.py`
- `mini_gpt/generate_attention_lm.py`

Stage 3 也抽取和修改了通用模块，供多个阶段复用：

- `mini_gpt/training.py`：新增通用 `load_config` 和 `estimate_loss`。
- `mini_gpt/utils.py`：新增 tokenizer 路径查找，并让 checkpoint 记录 `head_size`。

Stage 1 和 Stage 2 的原有命令保持可用。

模型流程变为：

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

## Stage 3 目标

1. 在 Stage 2 EmbeddingLanguageModel 的基础上加入单头 causal self-attention。
2. 理解 Q、K、V。
3. 理解 `QK^T / sqrt(d)`。
4. 理解 causal mask。
5. 理解 softmax attention weight。
6. 理解 `attention weight @ V`。
7. 支持 Mac MPS 小规模调试。
8. 支持 RTX 4090 24GB 较大配置训练。

## Stage 3 Mac 配置

Stage 3 的 Mac 配置用于快速调试和学习 attention shape，建议文件名：

```text
configs/single_head_attention_mac.yaml
```

建议参数：

```yaml
data:
  raw_text_path: data/raw.txt
  train_ratio: 0.9

model:
  block_size: 64
  n_embd: 128
  head_size: 128

train:
  batch_size: 32
  max_iters: 1000
  eval_interval: 100
  eval_iters: 20
  learning_rate: 0.001
  seed: 42
  device: auto

output:
  checkpoint_dir: checkpoints
  best_ckpt_name: single_head_attention_best.pt
  tokenizer_name: tokenizer.json
```

## Stage 3 4090 配置

Stage 3 的 4090 配置用于较大 block size、batch size 和训练步数，建议文件名：

```text
configs/single_head_attention_4090.yaml
```

建议参数：

```yaml
data:
  raw_text_path: data/full_corpus.txt
  train_ratio: 0.9

model:
  block_size: 256
  n_embd: 256
  head_size: 256

train:
  batch_size: 128
  max_iters: 10000
  eval_interval: 500
  eval_iters: 100
  learning_rate: 0.001
  seed: 42
  device: auto

output:
  checkpoint_dir: checkpoints
  best_ckpt_name: single_head_attention_4090_best.pt
  tokenizer_name: tokenizer_4090.json
```

## Stage 3 训练命令

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

## Stage 3 生成命令

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

## Stage 3 学习重点

1. Q、K、V 都来自同一个 hidden state `x`，但通过不同线性层得到。
2. `QK^T` 表示每个位置对其他位置的关注分数。
3. 除以 `sqrt(d)` 是为了让 attention score 的数值更稳定。
4. causal mask 让当前位置只能看见自己和之前的 token。
5. softmax 把被允许的位置转成概率权重。
6. `attention weight @ V` 得到融合上下文后的表示。
7. 本阶段只有一个 attention head，Multi-Head Attention 留到 Stage 4。

## Stage 3 注意力权重接口

`AttentionLanguageModel.forward(..., return_attention=True)` 会额外返回 attention weight：

```text
attention_weight shape: [batch_size, sequence_length, sequence_length]
```

这个接口用于后续做最小注意力可视化。

## Stage 3 禁止内容

Stage 3 不实现：

- Multi-Head Attention
- Transformer Block
- LayerNorm
- FFN
- LoRA
- SFT
- RAG
- `transformers` / `datasets` / `peft` / `accelerate` / `langchain` / `llama-index`

Stage 4 才进入 Multi-Head Attention。

## Stage 4：Multi-Head Causal Self-Attention

Stage 4 在 Stage 3 single-head causal self-attention 的基础上，扩展为多个 causal attention head 并行计算。

Stage 4 模型流程：

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

## Stage 4 目标

1. 在 Stage 3 单头 attention 的基础上实现多个 attention head。
2. 理解每个 head 都有自己的 Q、K、V。
3. 理解每个 head 都独立计算 `QK^T / sqrt(head_size)`。
4. 理解每个 head 都使用 causal mask。
5. 理解每个 head 都有自己的 softmax attention weight。
6. 理解多个 head 的输出如何 concat。
7. 理解 output projection 如何把 concat 输出映射回 `n_embd`。
8. concat 后继续交给 `lm_head` 做 next-token prediction。
9. 支持 Mac MPS 小规模调试。
10. 支持 RTX 4090 24GB 较大配置训练。

## Stage 4 和 Stage 3 的区别

Stage 3 只有一个 attention head：

```text
x -> single head -> lm_head
```

Stage 4 有多个 attention head：

```text
x -> head 0 / head 1 / ... / head n -> concat -> output projection -> lm_head
```

## Stage 4 Mac 配置

Stage 4 的 Mac 配置用于快速调试和学习 multi-head attention shape，建议文件名：

```text
configs/multi_head_lm_mac.yaml
```

建议参数：

```yaml
data:
  raw_text_path: data/raw.txt
  train_ratio: 0.9

model:
  block_size: 64
  n_embd: 128
  n_head: 4
  dropout: 0.0

train:
  batch_size: 32
  max_iters: 1000
  eval_interval: 100
  eval_iters: 20
  learning_rate: 0.001
  seed: 42
  device: auto

output:
  checkpoint_dir: checkpoints
  best_ckpt_name: multi_head_lm_best.pt
  tokenizer_name: tokenizer.json
```

## Stage 4 4090 配置

Stage 4 的 4090 配置用于较大 block size、batch size 和训练步数，建议文件名：

```text
configs/multi_head_lm_4090.yaml
```

建议参数：

```yaml
data:
  raw_text_path: data/computer_knowledge/raw/full_corpus.txt
  train_ratio: 0.9

model:
  block_size: 256
  n_embd: 256
  n_head: 8
  dropout: 0.0

train:
  batch_size: 64
  max_iters: 10000
  eval_interval: 500
  eval_iters: 100
  learning_rate: 0.001
  seed: 42
  device: auto

output:
  checkpoint_dir: checkpoints
  best_ckpt_name: multi_head_lm_4090_best.pt
  tokenizer_name: tokenizer_4090.json
```

## Stage 4 训练命令

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

## Stage 4 生成命令

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

## Stage 4 多头注意力可视化命令

```bash
python -m mini_gpt.visualize_multi_head_attention \
  --ckpt checkpoints/multi_head_lm_best.pt \
  --prompt "人工智能"
```

默认输出到：

```text
outputs/attention/
```

## Stage 4 学习重点

1. 多个 attention head 可以并行学习不同类型的 token 关系。
2. 每个 head 都是一套独立的 causal self-attention。
3. 每个 head 的 Q、K、V shape 是 `[batch_size, sequence_length, head_size]`。
4. 每个 head 的 attention score shape 是 `[batch_size, sequence_length, sequence_length]`。
5. causal mask shape 是 `[sequence_length, sequence_length]`。
6. 每个 head 的 attention weight shape 是 `[batch_size, sequence_length, sequence_length]`。
7. `head_size = n_embd // n_head`。
8. 多个 head 输出 concat 后 shape 回到 `[batch_size, sequence_length, n_embd]`。
9. output projection 继续保持 `[batch_size, sequence_length, n_embd]`。
10. 每个 head 都可以单独查看 attention weight。
11. Stage 4 不实现 Transformer Block，Stage 5 才进入 Transformer Block。

## Stage 4 禁止内容

Stage 4 不实现：

- Transformer Block
- LayerNorm
- FeedForward
- Residual Connection
- SFT
- LoRA
- RAG
- `transformers` / `datasets` / `peft` / `accelerate` / `langchain` / `llama-index`

Stage 5 才进入 Transformer Block。

## Stage 5：Transformer Block

Stage 5 在 Stage 4 Multi-Head Causal Self-Attention 的基础上，实现一个最小 Transformer Block。

Stage 5 模型流程：

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

## Stage 5 目标

1. 在 Stage 4 Multi-Head Causal Self-Attention 的基础上实现 Transformer Block。
2. 理解 LayerNorm 的作用。
3. 理解 Residual Connection 的作用。
4. 理解 FeedForward 的作用。
5. 理解 block 输入和输出 shape 都保持 `[batch_size, sequence_length, n_embd]`。
6. 继续做字符级 next-token prediction。
7. 支持 Mac MPS 小规模调试。
8. 支持 RTX 4090 24GB 较大配置训练。

## Stage 5 和 Stage 4 的区别

Stage 4 只实现 multi-head attention：

```text
x -> multi-head attention -> lm_head
```

Stage 5 把 multi-head attention 放进 Transformer Block：

```text
x -> LayerNorm -> multi-head attention -> residual
  -> LayerNorm -> feedforward -> residual
  -> lm_head
```

## Stage 5 Mac 配置

Stage 5 的 Mac 配置用于快速调试 Transformer Block，建议文件名：

```text
configs/transformer_block_lm_mac.yaml
```

建议参数：

```yaml
data:
  raw_text_path: data/raw.txt
  train_ratio: 0.9

model:
  block_size: 64
  n_embd: 128
  n_head: 4
  n_layer: 1
  dropout: 0.0

train:
  batch_size: 32
  max_iters: 1000
  eval_interval: 100
  eval_iters: 20
  learning_rate: 0.001
  seed: 42
  device: auto

output:
  checkpoint_dir: checkpoints
  best_ckpt_name: transformer_block_lm_best.pt
  tokenizer_name: tokenizer.json
```

## Stage 5 4090 配置

Stage 5 的 4090 配置用于较大 block size、batch size 和训练步数，建议文件名：

```text
configs/transformer_block_lm_4090.yaml
```

建议参数：

```yaml
data:
  raw_text_path: data/computer_knowledge/raw/full_corpus.txt
  train_ratio: 0.9

model:
  block_size: 256
  n_embd: 256
  n_head: 8
  n_layer: 1
  dropout: 0.1

train:
  batch_size: 64
  max_iters: 10000
  eval_interval: 500
  eval_iters: 100
  learning_rate: 0.001
  seed: 42
  device: auto

output:
  checkpoint_dir: checkpoints
  best_ckpt_name: transformer_block_lm_4090_best.pt
  tokenizer_name: tokenizer_4090.json
```

## Stage 5 训练命令

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

## Stage 5 生成命令

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

## Stage 5 可视化命令

Transformer Block attention 可视化：

```bash
python -m mini_gpt.visualize_transformer_block_attention \
  --ckpt checkpoints/transformer_block_lm_best.pt \
  --prompt "人工智能"
```

默认输出目录：

```text
outputs/transformer_block_attention/
```

## Stage 5 学习重点

1. Transformer Block 是 GPT 中反复堆叠的基本单元。
2. Stage 5 只实现一个 block，不实现完整 Decoder-only GPT。
3. LayerNorm 用来稳定子层输入。
4. Residual Connection 让输入可以绕过子层直接加到输出上。
5. FeedForward 对每个位置的 hidden state 做非线性变换。
6. Multi-Head Causal Self-Attention 复用 Stage 4 的实现。
7. block 输入和输出 shape 都是 `[batch_size, sequence_length, n_embd]`。
8. Stage 6 才进入完整 Decoder-only GPT。

## Stage 5 禁止内容

Stage 5 不实现：

- 完整 Decoder-only GPT
- 多层 Transformer Block 堆叠
- SFT
- LoRA
- RAG
- `transformers` / `datasets` / `peft` / `accelerate` / `langchain` / `llama-index`

Stage 6 才进入完整 Decoder-only GPT。

## Stage 6：完整 Decoder-only GPT

Stage 6 在 Stage 5 Transformer Block 的基础上，实现完整 Decoder-only GPT 主干模型。

Stage 6 模型流程：

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

## Stage 6 目标

1. 将 Stage 5 的 Transformer Block 堆叠成完整 Decoder-only GPT 主干。
2. 支持多个 Transformer Block。
3. 在所有 block 后加入 final LayerNorm。
4. 使用 `lm_head` 输出 vocab logits。
5. 继续做字符级 next-token prediction。
6. 支持 attention 权重可视化。
7. 支持模型参数统计。
8. 支持 Mac MPS 小规模调试。
9. 支持 RTX 4090 24GB 较大配置训练。
10. 支持最小 `temperature` 和 `top_k` 生成参数。

## Stage 6 和 Stage 5 的区别

Stage 5 只实现一个 Transformer Block：

```text
x -> Transformer Block -> lm_head
```

Stage 6 实现完整 GPT 主干：

```text
x -> Block 1 -> Block 2 -> ... -> Block n_layer -> final LayerNorm -> lm_head
```

## Stage 6 Mac 配置

Stage 6 的 Mac 配置用于快速调试完整 GPT 主干，建议文件名：

```text
configs/gpt_mac.yaml
```

建议参数：

```yaml
data:
  raw_text_path: data/raw.txt
  train_ratio: 0.9

model:
  block_size: 64
  n_embd: 128
  n_head: 4
  n_layer: 2
  dropout: 0.1

train:
  batch_size: 32
  max_iters: 1000
  eval_interval: 100
  eval_iters: 20
  learning_rate: 0.001
  weight_decay: 0.01
  seed: 42
  device: auto

generate:
  max_new_tokens: 200
  temperature: 1.0
  top_k: null

output:
  checkpoint_dir: checkpoints
  best_ckpt_name: gpt_best.pt
  tokenizer_name: tokenizer_gpt.json
```

## Stage 6 4090 配置

Stage 6 的 4090 配置用于较大 block size、embedding size、层数和训练步数，建议文件名：

```text
configs/gpt_4090.yaml
```

建议参数：

```yaml
data:
  raw_text_path: data/computer_knowledge/raw/full_corpus.txt
  train_ratio: 0.9

model:
  block_size: 256
  n_embd: 256
  n_head: 8
  n_layer: 4
  dropout: 0.1

train:
  batch_size: 64
  max_iters: 20000
  eval_interval: 500
  eval_iters: 100
  learning_rate: 0.0005
  weight_decay: 0.01
  seed: 42
  device: auto

generate:
  max_new_tokens: 500
  temperature: 0.9
  top_k: 50

output:
  checkpoint_dir: checkpoints
  best_ckpt_name: gpt_4090_best.pt
  tokenizer_name: tokenizer_gpt_4090.json
```

## Stage 6 训练命令

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

## Stage 6 生成命令

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

## Stage 6 可视化和统计命令

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

## Stage 6 学习重点

1. Decoder-only GPT 是多个 Transformer Block 的堆叠。
2. 每一层 block 的输入和输出 shape 都保持 `[batch_size, sequence_length, n_embd]`。
3. final LayerNorm 位于所有 block 之后、`lm_head` 之前。
4. `lm_head` 将 hidden state 映射为 `[batch_size, sequence_length, vocab_size]`。
5. attention 可视化需要区分 layer 和 head。
6. 参数量主要来自 embedding、attention、FeedForward、LayerNorm 和 `lm_head`。
7. `temperature` 控制 logits 缩放，数值越低越保守。
8. `top_k` 只保留分数最高的 k 个候选 token。
9. Stage 6 仍然使用字符级 tokenizer。
10. Stage 7 才进入更完整的生成策略优化。

## Stage 6 禁止内容

Stage 6 不实现：

- SFT
- LoRA
- RAG
- BPE tokenizer
- KV Cache
- Flash Attention
- `transformers` / `datasets` / `peft` / `accelerate` / `langchain` / `llama-index`

Stage 7 才进入生成策略优化。
