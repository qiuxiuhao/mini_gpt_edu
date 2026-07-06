# MiniGPT-edu

本项目用于从零实现一个教学版 GPT 语言模型，目标是帮助零基础学习者从项目实战出发理解大模型核心算法。

本项目会分阶段实现：

1. 字符级 Bigram 语言模型
2. Embedding
3. Single-Head Causal Self-Attention
4. Multi-Head Attention
5. Transformer Block
6. Decoder-only GPT
7. 生成策略
8. SFT
9. LoRA
10. RAG

## 当前阶段

Stage 3：Single-Head Causal Self-Attention。

Stage 1 字符级 Bigram 语言模型已完成。

Stage 2 Embedding 语言模型 + Mac/4090 双设备配置已完成。

Stage 3 当前只实现单头 causal self-attention，不实现 Multi-Head Attention、Transformer Block、LayerNorm、FFN、LoRA、SFT 或 RAG。

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

## 数据准备

默认小规模训练数据：

```text
data/raw.txt
```

4090 较大配置训练数据：

```text
data/full_corpus.txt
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
