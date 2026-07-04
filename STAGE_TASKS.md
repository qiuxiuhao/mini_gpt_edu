# 3. `STAGE_TASKS.md`：当前阶段任务

这个文件每个阶段都更新。现在第一阶段可以这样写：

```markdown
# STAGE_TASKS.md
# STAGE_TASKS.md

# 当前阶段：Stage 2 Embedding 语言模型

## 本阶段目标

当前项目已经完成 Stage 1：字符级 Bigram 语言模型。

现在进入 Stage 2：Embedding 语言模型 + 数据管线升级 + Mac/4090 双设备配置。

本阶段目标不是实现 Transformer，也不是实现 Attention，而是在 Bigram 的基础上理解 GPT 输入部分的核心结构：

1. token embedding
2. position embedding
3. lm_head
4. next-token prediction
5. Mac MPS / CUDA / CPU 自动设备选择
6. Mac 小规模调试配置
7. RTX 4090 24GB 较大训练配置

## 本阶段不要实现

本阶段禁止实现：

- Self-Attention
- Multi-Head Attention
- Transformer Block
- SFT
- LoRA
- RAG
- transformers / datasets / peft / accelerate / langchain / llama-index

## 本阶段需要新增的文件

```text
mini_gpt/
├── embedding_lm.py
├── train_embedding_lm.py
├── generate_embedding_lm.py
└── utils.py

configs/
├── embedding_lm_mac.yaml
└── embedding_lm_4090.yaml
```

## 本阶段验收命令

```bash
python -m mini_gpt.train_embedding_lm \
  --config configs/embedding_lm_mac.yaml \
  --max-iters 20

python -m mini_gpt.generate_embedding_lm \
  --ckpt checkpoints/embedding_lm_best.pt \
  --prompt "人工智能"
```

## Stage 2 和 Stage 3 的衔接

Stage 2 会得到 embedding 后的 hidden states：

```text
x shape: [batch_size, block_size, n_embd]
```

Stage 3 的 Causal Self-Attention 会接在这个 `x` 后面，再输出给 `lm_head`。

# 已完成阶段：Stage 1 字符级 Bigram 语言模型

## 阶段一目标

实现一个最小可运行的字符级 Bigram 语言模型，用于理解语言模型的基本训练流程。

本阶段不实现 Transformer，不实现 Self-Attention，不实现 LoRA，不实现 RAG。

## 阶段一需要实现的文件

```text
mini_gpt/
├── tokenizer.py
├── dataset.py
├── bigram.py
├── train_bigram.py
└── generate_bigram.py
```
