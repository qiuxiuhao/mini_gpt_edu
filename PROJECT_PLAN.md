# MiniGPT-edu 项目路线图

本项目目标是从零实现一个教学版 GPT 语言模型，帮助学习者从项目实战出发理解大模型核心算法。

本项目的目的旨在学习实现完整 Decoder-only GPT 主干，包括多层 Transformer Block 堆叠、final LayerNorm、`lm_head`、训练、生成、attention 可视化和模型参数统计。

当前项目已经完成到 Stage 6：完整 Decoder-only GPT。接下来后续学习可转至 minimind。

## 当前进度

- Stage 0：项目初始化，已完成。
- Stage 1：字符级 Bigram 语言模型，已完成。
- Stage 2：Embedding 语言模型 + Mac/4090 双设备配置，已完成。
- Stage 3：Single-Head Causal Self-Attention，已完成。
- Stage 4：Multi-Head Causal Self-Attention，已完成。
- Stage 5：Transformer Block，已完成。
- Stage 6：完整 Decoder-only GPT，已完成。

## Stage 0：项目初始化

状态：已完成。

目标：

- 创建项目结构。
- 配置 `AGENTS.md`。
- 配置 `requirements.txt`。
- 配置 `README.md`。
- 准备最小 `raw.txt` 数据。
- 准备基础配置文件。

产出：

- 可被学习者和 Codex 稳定理解的项目骨架。

## Stage 1：字符级 Bigram 语言模型

状态：已完成。

目标：

- 实现字符级 tokenizer。
- 构建字符表 vocab。
- 实现 encode / decode。
- 构造 next-token prediction 数据。
- 实现 `BigramLanguageModel`。
- 实现训练脚本。
- 实现生成脚本。

重点理解：

- token。
- vocab。
- token id。
- next-token prediction。
- logits。
- cross entropy。
- generation。

验收命令：

```bash
python -m mini_gpt.train_bigram --config configs/bigram.yaml
python -m mini_gpt.generate_bigram --ckpt checkpoints/bigram_best.pt --prompt "人工智能"
```

## Stage 2：Embedding 语言模型

状态：已完成。

目标：

- 在 Bigram 的基础上引入 token embedding。
- 引入 position embedding。
- 使用 `lm_head` 输出 vocab_size 维 logits。
- 继续使用 next-token prediction。
- 支持 Mac MPS 和 4090 CUDA 自动选择设备。
- 准备 Mac 小配置和 4090 较大配置。

重点理解：

- token id 不是向量。
- embedding table 是可学习查表矩阵。
- position embedding 用来提供位置信息。
- logits shape 为什么是 `[B, T, V]`。
- cross entropy 前为什么要 flatten。

验收命令：

```bash
python -m mini_gpt.train_embedding_lm --config configs/embedding_lm_mac.yaml --max-iters 20
python -m mini_gpt.generate_embedding_lm --ckpt checkpoints/embedding_lm_best.pt --prompt "人工智能"
```

## Stage 3：Single-Head Causal Self-Attention

状态：已完成。

目标：

- 在 Stage 2 `EmbeddingLanguageModel` 的基础上加入单头 causal self-attention。
- 理解 Q、K、V 的来源和 shape。
- 理解 `QK^T / sqrt(d)`。
- 理解 causal mask 如何阻止当前位置看到未来 token。
- 理解 softmax attention weight。
- 理解 `attention weight @ V`。
- 支持 Mac MPS 小规模调试。
- 支持 RTX 4090 24GB 较大配置训练。
- 抽取 `mini_gpt/training.py`，让 Stage 1/2/3 复用配置读取和 loss 评估。
- 在 `mini_gpt/utils.py` 中补充 Stage 3 checkpoint/tokenizer 通用工具。

验收命令：

```bash
python -m mini_gpt.train_attention_lm --config configs/single_head_attention_mac.yaml --max-iters 20
python -m mini_gpt.generate_attention_lm --ckpt checkpoints/single_head_attention_best.pt --prompt "人工智能"
```

## Stage 4：Multi-Head Causal Self-Attention

状态：已完成。

目标：

- 将 Stage 3 的 single-head attention 扩展为 multi-head attention。
- 理解多个 causal attention head 并行学习不同关系。
- 理解每个 head 都有自己的 Q、K、V。
- 理解每个 head 都使用 causal mask，不能看未来 token。
- 理解多个 head 的输出 concat。
- 理解 output projection 如何映射回 `n_embd`。
- 继续做字符级 next-token prediction。
- 支持 Mac MPS 小规模调试。
- 支持 RTX 4090 24GB 较大配置训练。

验收命令：

```bash
python -m mini_gpt.train_multi_head_lm --config configs/multi_head_lm_mac.yaml --max-iters 20
python -m mini_gpt.generate_multi_head_lm --ckpt checkpoints/multi_head_lm_best.pt --prompt "人工智能"
```

## Stage 5：Transformer Block

状态：已完成。

目标：

- 在 Stage 4 Multi-Head Causal Self-Attention 的基础上加入 Transformer Block。
- 引入 Residual Connection。
- 引入 LayerNorm。
- 引入 FeedForward。

验收命令：

```bash
python -m mini_gpt.train_transformer_block_lm --config configs/transformer_block_lm_mac.yaml --max-iters 20
python -m mini_gpt.generate_transformer_block_lm --ckpt checkpoints/transformer_block_lm_best.pt --prompt "人工智能"
```

## Stage 6：完整 Decoder-only GPT

状态：已完成。

目标：

- 将 Transformer Block 堆叠成完整 Decoder-only GPT。
- 支持多层 Transformer Block。
- 添加 final LayerNorm。
- 添加 `lm_head`。
- 继续做字符级 next-token prediction。
- 支持训练、生成、attention 可视化和模型参数统计。
- 支持最小 `temperature` 和 `top_k` 生成参数。
- 支持 Mac MPS 小规模调试。
- 支持 RTX 4090 24GB 较大配置训练。

验收命令：

```bash
python -m mini_gpt.train_gpt --config configs/gpt_mac.yaml --max-iters 20
python -m mini_gpt.generate_gpt --ckpt checkpoints/gpt_best.pt --prompt "人工智能"
python -m mini_gpt.visualize_gpt_attention --ckpt checkpoints/gpt_best.pt --prompt "人工智能"
python -m mini_gpt.model_summary --config configs/gpt_mac.yaml
```
