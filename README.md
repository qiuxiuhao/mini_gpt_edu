# MiniGPT-edu

本项目用于从零实现一个教学版 GPT 语言模型，目标是帮助零基础学习者从项目实战出发理解大模型核心算法。

本项目会分阶段实现：

1. 字符级 Bigram 语言模型
2. Embedding
3. Self-Attention
4. Multi-Head Attention
5. Transformer Block
6. Decoder-only GPT
7. 生成策略
8. SFT
9. LoRA
10. RAG

## 当前阶段

Stage 2：Embedding 语言模型。

## 环境配置

```bash
conda create -n mini_gpt_edu python=3.11 -y
conda activate mini_gpt_edu
pip install -r requirements.txt
```

## Stage 1 学习目标

当前阶段只实现字符级 Bigram 语言模型，用最小代码理解语言模型训练流程：

1. 字符如何变成 token id
2. 如何构造 next-token prediction 数据
3. logits 是什么
4. cross entropy loss 如何衡量预测错误
5. 如何通过训练降低 loss
6. 如何从 prompt 开始逐字符生成文本

本阶段不实现 Transformer、Self-Attention、Multi-Head Attention、LoRA 或 RAG。

## 数据准备

默认训练数据已经放在：

```text
data/raw.txt
```

这是一个很小的中文教学文本，适合快速跑通完整流程。

## 训练命令

```bash
python -m mini_gpt.train_bigram --config configs/bigram.yaml
```

训练完成后会保存：

```text
checkpoints/bigram_best.pt
checkpoints/tokenizer.json
```

## 快速测试命令

如果只想确认代码能跑通，可以覆盖训练步数：

```bash
python -m mini_gpt.train_bigram --config configs/bigram.yaml --max-iters 20
```

## 生成命令

```bash
python -m mini_gpt.generate_bigram \
  --ckpt checkpoints/bigram_best.pt \
  --prompt "人工智能"
```

## Stage 2：Embedding 语言模型

Stage 2 在 Stage 1 Bigram 的基础上，引入更接近 GPT 的输入结构：

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

## Stage 2 目标

本阶段继续使用字符级 tokenizer 和 next-token prediction，但模型不再让 token id 直接查出 logits，而是先学习 token embedding 和 position embedding。

模型显式包含：

1. `token_embedding_table = nn.Embedding(vocab_size, n_embd)`
2. `position_embedding_table = nn.Embedding(block_size, n_embd)`
3. `lm_head = nn.Linear(n_embd, vocab_size)`

本阶段不实现 Self-Attention、Transformer Block、Multi-Head Attention、SFT、LoRA 或 RAG。

## Stage 2 和 Stage 1 的区别

Stage 1 Bigram：

```text
token id -> logits
```

Stage 2 Embedding LM：

```text
token id -> token embedding
position id -> position embedding
token embedding + position embedding -> lm_head -> logits
```

Stage 2 的重点是理解 token id 不是向量，embedding table 才是可学习的向量查表矩阵。

## Stage 2 训练命令

MacBook Air M5 24GB 小规模训练：

```bash
python -m mini_gpt.train_embedding_lm --config configs/embedding_lm_mac.yaml
```

Mac 快速测试：

```bash
python -m mini_gpt.train_embedding_lm \
  --config configs/embedding_lm_mac.yaml \
  --max-iters 20
```

RTX 4090 24GB 较大配置训练：

```bash
python -m mini_gpt.train_embedding_lm --config configs/embedding_lm_4090.yaml
```

训练完成后会保存：

```text
checkpoints/embedding_lm_best.pt
checkpoints/tokenizer.json
```

## Stage 2 生成命令

```bash
python -m mini_gpt.generate_embedding_lm \
  --ckpt checkpoints/embedding_lm_best.pt \
  --prompt "人工智能"
```

## Stage 2 学习重点

1. token embedding：把 token id 查成可学习向量。
2. position embedding：给每个位置一个可学习向量。
3. logits：每个位置都输出对整个 vocab 的预测分数。
4. cross entropy：比较 logits 和下一个 token id。
5. device auto：优先 CUDA，其次 MPS，最后 CPU。

下一阶段将进入 Stage 3：Causal Self-Attention，把 attention 接在 embedding 之后。
