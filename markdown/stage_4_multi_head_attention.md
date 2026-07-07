# Stage 4：Multi-Head Causal Self-Attention

## 阶段定位

Stage 4 把 Stage 3 的单头注意力扩展成多头注意力。多个 head 并行工作，每个 head 都有自己的 Q、K、V 和 causal mask，最后把多个 head 的输出拼接起来，再经过输出投影。

## 本阶段核心升级

单头注意力更像“一个人从一个角度看上下文”，多头注意力更像“多个人从不同角度一起看上下文”。这种结构能让模型学习到更丰富的上下文关系。

## 代码位置索引

| 文件 | 类 / 函数 | 阅读重点 |
|---|---|---|
| `mini_gpt/attention.py` | `CausalSelfAttentionHead` | 单个 head 的计算单元 |
| `mini_gpt/attention.py` | `CausalSelfAttentionHead.__init__` | 定义单个 head 的 Q/K/V 和 mask |
| `mini_gpt/attention.py` | `CausalSelfAttentionHead.forward` | 单个 head 的 attention 计算 |
| `mini_gpt/attention.py` | `MultiHeadCausalSelfAttention` | 多头注意力整体 |
| `mini_gpt/attention.py` | `MultiHeadCausalSelfAttention.__init__` | 创建多个 head 和 output projection |
| `mini_gpt/attention.py` | `MultiHeadCausalSelfAttention.forward` | 并行执行多个 head、concat、projection |
| `mini_gpt/multi_head_lm.py` | `MultiHeadLanguageModel` | 多头注意力语言模型 |
| `mini_gpt/multi_head_lm.py` | `MultiHeadLanguageModel.__init__` | embedding、多头注意力、lm_head |
| `mini_gpt/multi_head_lm.py` | `MultiHeadLanguageModel.forward` | 前向传播并可选返回 attention |
| `mini_gpt/multi_head_lm.py` | `MultiHeadLanguageModel.generate` | 多头模型生成文本 |
| `mini_gpt/train_multi_head_lm.py` | `main` | Stage 4 训练入口 |
| `mini_gpt/generate_multi_head_lm.py` | `main` | Stage 4 生成入口 |
| `mini_gpt/visualize_multi_head_attention.py` | `save_head_heatmap` | 保存单个 head 的注意力热力图 |
| `mini_gpt/visualize_multi_head_attention.py` | `main` | 加载模型并保存所有 head 的热力图 |
| `mini_gpt/utils.py` | `save_checkpoint` | Stage 4 额外保存 `n_head` |

## 关键 shape

| 张量 | shape |
|---|---|
| `x` | `[B, T, C]` |
| 单个 head 的 `q/k/v` | `[B, T, head_size]` |
| 单个 head 的 attention weight | `[B, T, T]` |
| 多个 head 拼接后 | `[B, T, C]` |
| output projection 后 | `[B, T, C]` |
| logits | `[B, T, V]` |

## 本阶段你应该掌握

- 为什么 `n_embd` 必须能被 `n_head` 整除
- `head_size = n_embd / n_head` 的含义
- 多个 head 如何并行工作
- concat 后为什么还要 output projection
- attention 可视化如何帮助理解不同 head 的分工
