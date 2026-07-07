# AGENTS.md

## 项目定位

本项目名为 `mini_gpt_edu`。

目标是面向零基础学生，从项目实战出发，逐步理解并实现大语言模型的核心算法。

本项目不是为了训练大规模模型，也不是为了追求最强效果，而是为了教学、理解和简历展示。

本项目的目的旨在学习实现完整 Decoder-only GPT 主干，包括多层 Transformer Block 堆叠、final LayerNorm、`lm_head`、训练、生成、attention 可视化和模型参数统计。

当前项目已经完成到 Stage 6：完整 Decoder-only GPT。接下来后续学习可转至 minimind。

## 已完成内容

1. 字符级 tokenizer。
2. Bigram 语言模型。
3. Embedding。
4. Causal Self-Attention。
5. Multi-Head Attention。
6. Transformer Block。
7. Decoder-only GPT。
8. 训练与生成。
9. 注意力可视化。
10. 模型参数统计。

## 最终阶段状态

当前已经完成：

- Stage 1：字符级 Bigram 语言模型。
- Stage 2：Embedding 语言模型 + Mac/4090 双设备配置。
- Stage 3：Single-Head Causal Self-Attention。
- Stage 4：Multi-Head Causal Self-Attention。
- Stage 5：Transformer Block。
- Stage 6：完整 Decoder-only GPT。

Stage 6 的实现范围：

- 完整 Decoder-only GPT 主干模型。
- 多层 Transformer Block 堆叠。
- final LayerNorm。
- `lm_head`。
- 训练。
- 生成。
- attention 可视化。
- 模型参数统计。
- 最小 `temperature` 和 `top_k` 生成参数。
- Mac MPS 小规模调试。
- RTX 4090 24GB 较大配置训练。

## 运行环境

目标设备：

- MacBook Air M4。
- RTX 4090 24GB。
- 24GB 内存。
- Apple Silicon。
- 优先支持 CUDA、MPS 和 CPU 自动选择。

Python 版本：

- Python 3.11+。

主要依赖：

- torch。
- numpy。
- matplotlib。
- pyyaml。
- tqdm。

不要默认引入大型复杂依赖。

## 代码风格

请保持代码适合零基础学生阅读：

1. 函数和类不要写得太长。
2. 每个核心函数都要有清晰 docstring。
3. 关键张量形状必须写在注释里。
4. 变量名要直观，例如 `token_ids`、`logits`、`loss`、`batch_size`、`block_size`。
5. 不要过度封装。
6. 不要为了工程复杂度而创建太多抽象层。
7. 优先保证清晰，而不是追求性能极致。

## 教学要求

每个核心模块应在代码注释中说明：

1. 这个模块解决什么问题。
2. 输入是什么。
3. 输出是什么。
4. 张量 shape 是什么。
5. 它在语言模型中的位置是什么。
6. 和前一个阶段有什么关系。

例如：

```python
# x shape: [batch_size, block_size]
# logits shape: [batch_size, block_size, vocab_size]
```
