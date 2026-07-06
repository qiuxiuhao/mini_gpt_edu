# PROJECT_PLAN.md

# MiniGPT-edu 项目路线图

本项目目标是从零实现一个教学版 GPT 语言模型，帮助学习者从项目实战出发理解大模型核心算法。

每次进入新阶段，必须优先阅读 `STAGE_TASKS.md`。`STAGE_TASKS.md` 中的当前阶段任务优先级高于 `README.md` 中的历史说明。

## 当前进度

- Stage 0：项目初始化，已完成。
- Stage 1：字符级 Bigram 语言模型，已完成。
- Stage 2：Embedding 语言模型 + Mac/4090 双设备配置，已完成。
- Stage 3：Single-Head Causal Self-Attention，已完成。
- Stage 4：Multi-Head Causal Self-Attention，当前进行中。
- Stage 5：Transformer Block，下一阶段实现。

## Stage 0：项目初始化

目标：

- 创建项目结构
- 配置 `AGENTS.md`
- 配置 `requirements.txt`
- 配置 `README.md`
- 准备最小 `raw.txt` 数据
- 准备基础配置文件

产出：

- 可被 Codex 稳定理解的项目骨架

## Stage 1：字符级 Bigram 语言模型

状态：已完成。

目标：

- 实现字符级 tokenizer
- 构建字符表 vocab
- 实现 encode / decode
- 构造 next-token prediction 数据
- 实现 `BigramLanguageModel`
- 实现训练脚本
- 实现生成脚本

重点理解：

- token
- vocab
- token id
- next-token prediction
- logits
- cross entropy
- generation

验收命令：

```bash
python -m mini_gpt.train_bigram --config configs/bigram.yaml
python -m mini_gpt.generate_bigram --ckpt checkpoints/bigram_best.pt --prompt "人工智能"
```

## Stage 2：Embedding 语言模型

状态：已完成。

目标：

- 在 Bigram 的基础上引入 token embedding
- 引入 position embedding
- 使用 lm_head 输出 vocab_size 维 logits
- 继续使用 next-token prediction
- 支持 Mac MPS 和 4090 CUDA 自动选择设备
- 准备 Mac 小配置和 4090 较大配置

重点理解：

- token id 不是向量
- embedding table 是可学习查表矩阵
- position embedding 用来提供位置信息
- logits shape 为什么是 `[B, T, V]`
- cross entropy 前为什么要 flatten

本阶段不实现：

- Self-Attention
- Transformer
- LoRA
- RAG

验收命令：

```bash
python -m mini_gpt.train_embedding_lm --config configs/embedding_lm_mac.yaml --max-iters 20
python -m mini_gpt.generate_embedding_lm --ckpt checkpoints/embedding_lm_best.pt --prompt "人工智能"
```

## Stage 3：Single-Head Causal Self-Attention

状态：已完成。

目标：

- 在 Stage 2 `EmbeddingLanguageModel` 的基础上加入单头 causal self-attention
- 理解 Q、K、V 的来源和 shape
- 理解 `QK^T / sqrt(d)`
- 理解 causal mask 如何阻止当前位置看到未来 token
- 理解 softmax attention weight
- 理解 `attention weight @ V`
- 支持 Mac MPS 小规模调试
- 支持 RTX 4090 24GB 较大配置训练
- 抽取 `mini_gpt/training.py`，让 Stage 1/2/3 复用配置读取和 loss 评估
- 在 `mini_gpt/utils.py` 中补充 Stage 3 checkpoint/tokenizer 通用工具

本阶段只实现 single-head causal self-attention。

本阶段不实现：

- Multi-Head Attention
- Transformer Block
- LayerNorm
- FFN
- LoRA
- SFT
- RAG
- `transformers` / `datasets` / `peft` / `accelerate` / `langchain` / `llama-index`

计划验收命令：

```bash
python -m mini_gpt.train_attention_lm --config configs/single_head_attention_mac.yaml --max-iters 20
python -m mini_gpt.generate_attention_lm --ckpt checkpoints/single_head_attention_best.pt --prompt "人工智能"
```

## Stage 4：Multi-Head Causal Self-Attention

状态：当前进行中。

目标：

- 将 Stage 3 的 single-head attention 扩展为 multi-head attention
- 理解多个 causal attention head 并行学习不同关系
- 理解每个 head 都有自己的 Q、K、V
- 理解每个 head 都使用 causal mask，不能看未来 token
- 理解多个 head 的输出 concat
- 理解 output projection 如何映射回 n_embd
- 继续做字符级 next-token prediction
- 支持 Mac MPS 小规模调试
- 支持 RTX 4090 24GB 较大配置训练

本阶段只实现 Multi-Head Causal Self-Attention。

本阶段不实现：

- Transformer Block
- LayerNorm
- FeedForward
- Residual Connection
- SFT
- LoRA
- RAG
- `transformers` / `datasets` / `peft` / `accelerate` / `langchain` / `llama-index`

计划验收命令：

```bash
python -m mini_gpt.train_multi_head_lm --config configs/multi_head_lm_mac.yaml --max-iters 20
python -m mini_gpt.generate_multi_head_lm --ckpt checkpoints/multi_head_lm_best.pt --prompt "人工智能"
```

## Stage 5：Transformer Block

状态：下一阶段实现。

目标：

- 在 Stage 4 Multi-Head Causal Self-Attention 的基础上加入 Transformer Block
- 引入 Residual Connection
- 引入 LayerNorm
- 引入 FeedForward

Stage 5 之前不要实现 Transformer Block。
