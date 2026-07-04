# 2. `PROJECT_PLAN.md`：整个项目路线图

这个是让 Codex 永远知道：我们不是随便写代码，而是在一步步做一个教学版 MiniGPT。

```markdown
# PROJECT_PLAN.md

# MiniGPT-edu 项目路线图

本项目目标是从零实现一个教学版 GPT 语言模型，帮助学习者从项目实战出发理解大模型核心算法。

## Stage 0：项目初始化

目标：

- 创建项目结构
- 配置 AGENTS.md
- 配置 requirements.txt
- 配置 README.md
- 准备最小 raw.txt 数据
- 准备基础配置文件

产出：

- 可被 Codex 稳定理解的项目骨架

---

## Stage 1：字符级 Bigram 语言模型

目标：

- 实现字符级 tokenizer
- 构建字符表 vocab
- 实现 encode / decode
- 构造 next-token prediction 数据
- 实现 BigramLanguageModel
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

## 当前进度

- Stage 1：字符级 Bigram 语言模型，已完成。
- Stage 2：Embedding 语言模型，当前正在实现与验收。
- Stage 3：Self-Attention，下一阶段实现。

## Stage 2：Embedding 语言模型

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

# 每次进入新阶段，必须优先阅读 STAGE_TASKS.md。STAGE_TASKS.md 中的当前阶段任务优先级高于 README 中的历史说明。
