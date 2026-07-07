# Stage 2：Embedding 语言模型

## 阶段定位

Stage 2 从 Bigram 升级到 Embedding Language Model。模型不再直接把当前 token 映射成下一个 token 的 logits，而是先把 token id 变成向量表示，再加入位置向量，最后通过 `lm_head` 映射到词表大小。

## 本阶段核心升级

Stage 1 更像查一张 Bigram 表；Stage 2 开始引入“表示学习”。Token Embedding 告诉模型“这个 token 是什么”，Position Embedding 告诉模型“这个 token 在第几个位置”。

## 代码位置索引

| 文件 | 类 / 函数 | 阅读重点 |
|---|---|---|
| `mini_gpt/embedding_lm.py` | `EmbeddingLanguageModel` | Stage 2 模型整体 |
| `mini_gpt/embedding_lm.py` | `EmbeddingLanguageModel.__init__` | 定义 token embedding、position embedding、lm_head |
| `mini_gpt/embedding_lm.py` | `EmbeddingLanguageModel.forward` | token_emb + pos_emb，再输出 logits |
| `mini_gpt/embedding_lm.py` | `EmbeddingLanguageModel.generate` | 使用当前序列自回归生成 token |
| `mini_gpt/train_embedding_lm.py` | `main` | Stage 2 训练入口 |
| `mini_gpt/generate_embedding_lm.py` | `main` | Stage 2 生成入口 |
| `mini_gpt/training.py` | `load_config` | 读取 YAML 配置 |
| `mini_gpt/training.py` | `estimate_loss` | 统一评估 train / val loss |
| `mini_gpt/utils.py` | `set_seed` | 固定随机种子 |
| `mini_gpt/utils.py` | `get_device` | 自动选择训练设备 |
| `mini_gpt/utils.py` | `ensure_dir` | 确保目录存在 |
| `mini_gpt/utils.py` | `save_checkpoint` | 保存模型、配置和 tokenizer 路径 |
| `mini_gpt/utils.py` | `load_checkpoint` | 加载 checkpoint |
| `mini_gpt/utils.py` | `resolve_tokenizer_path` | 找到 checkpoint 对应 tokenizer |

## 关键 shape

| 张量 | 含义 | shape |
|---|---|---|
| `idx` | 输入 token ids | `[B, T]` |
| `token_emb` | token embedding 后的表示 | `[B, T, C]` |
| `pos_emb` | position embedding 后的位置表示 | `[T, C]` |
| `x` | token 与 position 相加后的表示 | `[B, T, C]` |
| `logits` | 每个位置对词表的预测得分 | `[B, T, V]` |

## 关键流程

1. 输入 `idx`。
2. `token_embedding_table` 把每个 token id 变成向量。
3. `position_embedding_table` 生成位置向量。
4. token 向量和位置向量相加，得到带位置信息的表示。
5. `lm_head` 把隐藏表示映射到词表大小。
6. 训练时计算 cross entropy。
7. 生成时取最后一个位置的 logits，采样下一个 token。

## 本阶段你应该掌握

- 为什么 token 需要变成向量
- Position Embedding 的作用
- `lm_head` 的作用
- `[B, T, C]` 和 `[B, T, V]` 的区别
- 为什么 Stage 2 还没有真正的上下文交互
