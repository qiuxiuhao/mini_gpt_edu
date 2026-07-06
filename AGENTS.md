# AGENTS.md

## 项目定位

本项目名为 `mini_gpt_edu`。

目标是面向零基础学生，从项目实战出发，逐步理解并实现大语言模型的核心算法。

本项目不是为了训练大规模模型，也不是为了追求最强效果，而是为了教学、理解和简历展示。

最终项目希望覆盖：

1. 字符级 tokenizer
2. Bigram 语言模型
3. Embedding
4. Causal Self-Attention
5. Multi-Head Attention
6. Transformer Block
7. Decoder-only GPT
8. 训练与生成
9. 注意力可视化
10. SFT 指令微调
11. LoRA 微调
12. RAG 知识库问答

## 当前阶段状态

当前已经完成：

- Stage 1：字符级 Bigram 语言模型
- Stage 2：Embedding 语言模型 + Mac/4090 双设备配置
- Stage 3：Single-Head Causal Self-Attention

当前进入：

- Stage 4：Multi-Head Causal Self-Attention

Stage 4 的实现边界非常重要：

- 只在 Stage 3 single-head causal self-attention 的基础上实现 multi-head causal self-attention。
- 只学习多个 attention head 并行计算、每个 head 的 Q/K/V、每个 head 的 causal mask、每个 head 的 softmax attention weight、多个 head 输出 concat 和 output projection。
- Stage 4 新增或修改的代码要用 docstring 或关键注释标明 `Stage 4 新增`、`Stage 4 修改` 或 `Stage 4 抽取`。
- 支持 Mac MPS 小规模调试。
- 支持 RTX 4090 24GB 较大配置训练。

Stage 4 禁止实现：

- Transformer Block
- LayerNorm
- FeedForward
- Residual Connection
- LoRA
- SFT
- RAG
- `transformers` / `datasets` / `peft` / `accelerate` / `langchain` / `llama-index`

Stage 5 才实现 Transformer Block。Stage 4 不要提前实现 Stage 5 的内容。

## 运行环境

目标设备：

- MacBook Air M4
- RTX 4090 24GB
- 24GB 内存
- Apple Silicon
- 优先支持 CUDA、MPS 和 CPU 自动选择

Python 版本：

- Python 3.11+

主要依赖：

- torch
- numpy
- matplotlib
- pyyaml
- tqdm

不要默认引入大型复杂依赖。

除非用户明确要求，否则不要使用：

- transformers
- datasets
- accelerate
- peft
- langchain
- llama-index

这些库可以放在后续阶段，不要在 Stage 4 引入。

## 代码风格

请保持代码适合零基础学生阅读：

1. 函数和类不要写得太长。
2. 每个核心函数都要有清晰 docstring。
3. 关键张量形状必须写在注释里。
4. 变量名要直观，例如 `token_ids`, `logits`, `loss`, `batch_size`, `block_size`。
5. 不要过度封装。
6. 不要为了工程复杂度而创建太多抽象层。
7. 当前阶段优先保证清晰，而不是追求性能极致。

## 教学要求

每次实现新模块时，请在代码注释中说明：

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
