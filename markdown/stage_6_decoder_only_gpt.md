# Stage 6：完整 Decoder-only GPT

## 阶段定位

Stage 6 在 Stage 5 的单个 Transformer Block 基础上，堆叠多个 Transformer Block，并在最后加入 final LayerNorm 和 `lm_head`，形成完整 Decoder-only GPT 主干。

## 本阶段核心升级

Stage 5 只有一个 block；Stage 6 把多个 block 按顺序堆叠。每个 block 都会逐层加工 hidden state，但每层输入和输出 shape 都保持 `[B, T, C]`。最后通过 final LayerNorm 稳定最终表示，再用 `lm_head` 输出词表 logits。

## 代码位置索引

| 文件 | 类 / 函数 | 阅读重点 |
|---|---|---|
| `mini_gpt/gpt.py` | `GPTLanguageModel` | 完整 Decoder-only GPT 主干 |
| `mini_gpt/gpt.py` | `GPTLanguageModel.__init__` | 定义 embedding、多个 block、final LayerNorm、lm_head |
| `mini_gpt/gpt.py` | `GPTLanguageModel.forward` | 多层 block 前向传播，并可选返回所有层 attention |
| `mini_gpt/gpt.py` | `GPTLanguageModel.generate` | 支持 temperature 和 top_k 的自回归生成 |
| `mini_gpt/train_gpt.py` | `count_parameters` | 统计模型总参数量 |
| `mini_gpt/train_gpt.py` | `main` | Stage 6 GPT 训练入口 |
| `mini_gpt/generate_gpt.py` | `main` | 从 checkpoint 加载 GPT 并生成文本 |
| `mini_gpt/model_summary.py` | `count_parameters` | 统计任意模块参数量 |
| `mini_gpt/model_summary.py` | `main` | 打印模型结构和各模块参数量 |
| `mini_gpt/visualize_gpt_attention.py` | `save_attention_heatmap` | 保存某一层某个 head 的注意力热力图 |
| `mini_gpt/visualize_gpt_attention.py` | `main` | 加载 checkpoint，逐层逐头保存 attention 图 |

## GPT 主干流程

1. 输入 `idx`，shape 为 `[B, T]`。
2. Token Embedding 得到 `[B, T, C]`。
3. Position Embedding 得到 `[T, C]`，并与 token embedding 相加。
4. 经过 `n_layer` 个 Transformer Block。
5. 每个 block 输入输出都保持 `[B, T, C]`。
6. 经过 final LayerNorm，得到最终 hidden state。
7. `lm_head` 把 `[B, T, C]` 映射成 `[B, T, V]`。
8. 训练时将 logits 和 targets 展平后计算 Cross Entropy。

## 生成流程

1. 输入 prompt。
2. tokenizer 把 prompt 编码为 token id。
3. 如果序列太长，只保留最后 `block_size` 个 token 作为上下文。
4. GPT forward 输出 logits。
5. 只取最后一个位置的 logits。
6. 使用 temperature 控制随机性。
7. 使用 top_k 限制候选 token 范围。
8. softmax 得到概率。
9. multinomial 采样 next_id。
10. 把 next_id 拼回 idx。
11. 重复直到生成足够多 token。

## 注意力可视化与参数统计

- `visualize_gpt_attention.py` 可以查看不同 layer、不同 head 的 attention weight。
- 单层 attention weight 的 shape 是 `[B, H, T, T]`。
- 全部层的 attention 是 `L` 个 `[B, H, T, T]`。
- `model_summary.py` 可以统计总参数量以及 token embedding、position embedding、blocks、final LayerNorm、lm_head 各自的参数量。

## 本阶段你应该掌握

- 完整 Decoder-only GPT 的主干组成
- 多层 Transformer Block 如何堆叠
- final LayerNorm 的位置和作用
- `temperature` 和 `top_k` 对生成的影响
- attention 可视化如何扩展到 layer 和 head 两个维度
- 模型参数主要来自 embedding、blocks 和 lm_head
