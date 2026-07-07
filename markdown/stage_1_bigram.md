# Stage 1：字符级 Bigram 语言模型

## 阶段定位

Stage 1 用最小的字符级 Bigram 语言模型解释语言模型的基本闭环：文本变成 token id，模型根据当前 token 预测下一个 token，使用交叉熵计算 loss，再用优化器更新参数。

## 本阶段核心思想

Bigram 模型只看当前字符，不看更长上下文。它学习的是“当前 token 到下一个 token 的概率表”。虽然能力很弱，但它能帮助你理解 token、vocab、logits、cross entropy、checkpoint、generate 这些后续 GPT 都会继续使用的概念。

## 代码位置索引

| 文件 | 类 / 函数 | 阅读重点 |
|---|---|---|
| `mini_gpt/tokenizer.py` | `CharTokenizer` | 字符级 tokenizer 的整体结构 |
| `mini_gpt/tokenizer.py` | `CharTokenizer.__init__` | 保存字符表、`stoi`、`itos` |
| `mini_gpt/tokenizer.py` | `CharTokenizer.from_text` | 从原始文本构建字符表 |
| `mini_gpt/tokenizer.py` | `CharTokenizer.vocab_size` | 返回词表大小 |
| `mini_gpt/tokenizer.py` | `CharTokenizer.encode` | 文本转 token id |
| `mini_gpt/tokenizer.py` | `CharTokenizer.decode` | token id 转文本 |
| `mini_gpt/tokenizer.py` | `CharTokenizer.save` | 保存 tokenizer |
| `mini_gpt/tokenizer.py` | `CharTokenizer.load` | 加载 tokenizer |
| `mini_gpt/dataset.py` | `read_text` | 读取原始训练文本 |
| `mini_gpt/dataset.py` | `train_val_split` | 划分训练集和验证集 |
| `mini_gpt/dataset.py` | `get_batch` | 构造 x/y，其中 y 是 x 右移一位 |
| `mini_gpt/bigram.py` | `BigramLanguageModel` | Bigram 模型整体 |
| `mini_gpt/bigram.py` | `BigramLanguageModel.__init__` | 定义 token embedding 表 |
| `mini_gpt/bigram.py` | `BigramLanguageModel.forward` | 根据 idx 得到 logits，并可选计算 loss |
| `mini_gpt/bigram.py` | `BigramLanguageModel.generate` | 自回归生成 token id |
| `mini_gpt/train_bigram.py` | `load_config` | 读取 YAML 配置 |
| `mini_gpt/train_bigram.py` | `get_device` | 选择 CPU / MPS / CUDA |
| `mini_gpt/train_bigram.py` | `estimate_loss` | 评估 train / val loss |
| `mini_gpt/train_bigram.py` | `save_checkpoint` | 保存最佳模型 |
| `mini_gpt/train_bigram.py` | `main` | 训练主入口 |
| `mini_gpt/generate_bigram.py` | `main` | 从 checkpoint 加载并生成文本 |

## 关键流程

1. `read_text` 读取文本。
2. `CharTokenizer.from_text` 构建字符表。
3. `encode` 把文本变成 token id 序列。
4. `get_batch` 随机截取一段序列作为 x，并把右移一位作为 y。
5. `BigramLanguageModel.forward` 输出 logits。
6. 如果传入 targets，就计算交叉熵 loss。
7. 训练脚本反向传播并更新参数。
8. `generate` 每次采样一个新 token，并拼回序列末尾。

## 本阶段你应该掌握

- token 和 vocab 的含义
- logits 与概率的区别
- x/y 右移一位的 next-token prediction 训练方式
- 为什么训练时要计算 cross entropy
- 为什么生成时是一个 token 一个 token 往后接
