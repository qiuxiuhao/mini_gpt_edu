# Stage 3：Single-Head Causal Self-Attention

## 阶段定位

Stage 3 在 Embedding LM 的基础上加入单头因果自注意力。这个阶段的重点是让每个位置不再只看自己的 token 表示，而是可以读取左侧历史 token 的信息。

## 本阶段核心升级

Stage 2 只有 token embedding 和 position embedding，每个位置主要独立预测。Stage 3 加入 Causal Self-Attention 后，当前位置可以根据 Q/K/V 和 attention weight 选择性地读取历史上下文，但不能看未来 token。

## 代码位置索引

| 文件 | 类 / 函数 | 阅读重点 |
|---|---|---|
| `mini_gpt/attention.py` | `SingleHeadCausalSelfAttention` | 单头因果自注意力整体 |
| `mini_gpt/attention.py` | `SingleHeadCausalSelfAttention.__init__` | 定义 key、query、value、causal mask |
| `mini_gpt/attention.py` | `SingleHeadCausalSelfAttention.forward` | 计算 Q/K/V、attention score、mask、softmax、weighted sum |
| `mini_gpt/attention_lm.py` | `AttentionLanguageModel` | Embedding LM + 单头注意力的语言模型 |
| `mini_gpt/attention_lm.py` | `AttentionLanguageModel.__init__` | 定义 embedding、attention、lm_head |
| `mini_gpt/attention_lm.py` | `AttentionLanguageModel.forward` | embedding 后进入 attention，再输出 logits |
| `mini_gpt/attention_lm.py` | `AttentionLanguageModel.generate` | 单头注意力模型的自回归生成 |
| `mini_gpt/train_attention_lm.py` | `main` | Stage 3 训练入口 |
| `mini_gpt/generate_attention_lm.py` | `main` | Stage 3 生成入口 |
| `mini_gpt/training.py` | `load_config` | 抽取出的通用配置读取函数 |
| `mini_gpt/training.py` | `estimate_loss` | 抽取出的通用 loss 评估函数 |
| `mini_gpt/utils.py` | `get_device` | 通用设备选择 |
| `mini_gpt/utils.py` | `save_checkpoint` | 保存模型元数据 |
| `mini_gpt/utils.py` | `resolve_tokenizer_path` | 兼容不同阶段 checkpoint 的 tokenizer 路径 |

## Attention 计算逻辑

1. 输入 `x` 的 shape 是 `[B, T, C]`。
2. 线性映射得到 Q、K、V。
3. 计算注意力分数 `QK^T / sqrt(head_size)`。
4. 使用 causal mask 把未来位置屏蔽掉。
5. 对分数做 softmax 得到 attention weight。
6. 用 attention weight 加权求和 V，得到上下文表示。
7. 输出仍保持 `[B, T, head_size]` 或映射回模型需要的维度。

## 本阶段你应该掌握

- Q、K、V 分别代表什么
- 为什么要除以 `sqrt(head_size)`
- causal mask 为什么能防止偷看未来
- attention weight 的 shape 为什么是 `[B, T, T]`
- 训练和生成为什么都必须遵守“只能看左边”的规则
