# Stage 5：Transformer Block

## 阶段定位

Stage 5 在 Stage 4 的 Multi-Head Causal Self-Attention 外面，加入 LayerNorm、FeedForward 和 Residual Connection，形成一个最小 Transformer Block。这个阶段的目标是理解 GPT block 的基本结构。

## 本阶段核心升级

Stage 4 主要解决“多个 head 如何看上下文”。Stage 5 进一步加入完整 block 结构：先 LayerNorm，再 Attention，再残差相加；再 LayerNorm，再 FeedForward，再残差相加。

## 代码位置索引

| 文件 | 类 / 函数 | 阅读重点 |
|---|---|---|
| `mini_gpt/transformer_block.py` | `FeedForward` | Transformer Block 内部的逐位置前馈网络 |
| `mini_gpt/transformer_block.py` | `FeedForward.__init__` | Linear、GELU、Linear、Dropout 的结构 |
| `mini_gpt/transformer_block.py` | `FeedForward.forward` | 对每个位置独立做非线性加工 |
| `mini_gpt/transformer_block.py` | `TransformerBlock` | 一个 pre-LN Transformer Block |
| `mini_gpt/transformer_block.py` | `TransformerBlock.__init__` | 定义 ln1、attn、ln2、ffn |
| `mini_gpt/transformer_block.py` | `TransformerBlock.forward` | 先 LN + Attention + Residual，再 LN + FFN + Residual |
| `mini_gpt/transformer_block_lm.py` | `TransformerBlockLanguageModel` | 带一个 Transformer Block 的语言模型 |
| `mini_gpt/transformer_block_lm.py` | `TransformerBlockLanguageModel.__init__` | embedding、单个 block、lm_head |
| `mini_gpt/transformer_block_lm.py` | `TransformerBlockLanguageModel.forward` | embedding 后进入 block，再输出 logits |
| `mini_gpt/transformer_block_lm.py` | `TransformerBlockLanguageModel.generate` | Transformer Block LM 的生成逻辑 |
| `mini_gpt/train_transformer_block_lm.py` | `main` | Stage 5 训练入口 |
| `mini_gpt/generate_transformer_block_lm.py` | `main` | Stage 5 生成入口 |
| `mini_gpt/visualize_transformer_block_attention.py` | `save_head_heatmap` | 保存 block 内每个 head 的注意力图 |
| `mini_gpt/visualize_transformer_block_attention.py` | `main` | 加载 checkpoint 并可视化 attention |
| `mini_gpt/utils.py` | `save_checkpoint` | Stage 5 额外保存 `n_layer` |

## Transformer Block 公式

- 第一段：`x = x + Attention(LayerNorm(x))`
- 第二段：`x = x + FeedForward(LayerNorm(x))`

## 关键 shape

| 张量 | shape |
|---|---|
| `x` | `[B, T, C]` |
| `ln1_out` | `[B, T, C]` |
| `attention_out` | `[B, T, C]` |
| `ln2_out` | `[B, T, C]` |
| `feed_forward_out` | `[B, T, C]` |
| `block_out` | `[B, T, C]` |
| `logits` | `[B, T, V]` |

## 本阶段你应该掌握

- LayerNorm 为什么能稳定训练
- Residual Connection 为什么能保留原信息并改善梯度传播
- FeedForward 为什么是逐位置加工
- Pre-LN Transformer Block 的结构顺序
- 为什么 block 输入和输出 shape 必须保持 `[B, T, C]`
