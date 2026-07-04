# 3. `STAGE_TASKS.md`：当前阶段任务

这个文件每个阶段都更新。现在第一阶段可以这样写：

```markdown
# STAGE_TASKS.md

# 当前阶段：Stage 1 字符级 Bigram 语言模型

## 本阶段目标

实现一个最小可运行的字符级 Bigram 语言模型，用于理解语言模型的基本训练流程。

本阶段不实现 Transformer，不实现 Self-Attention，不实现 LoRA，不实现 RAG。

## 本阶段需要实现的文件

```text
mini_gpt/
├── tokenizer.py
├── dataset.py
├── bigram.py
├── train_bigram.py
└── generate_bigram.py