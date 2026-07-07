# MiniGPT-edu

MiniGPT-edu 是一个从零实现教学版 GPT 语言模型的项目，目标是帮助零基础学习者通过项目实战理解大模型核心算法。

项目按阶段推进，从字符级 Bigram 语言模型逐步扩展到 Embedding、Causal Self-Attention、Multi-Head Attention、Transformer Block、Decoder-only GPT，并在后续阶段继续进入生成策略、SFT、LoRA 和 RAG。

## 阶段总览

| 阶段 | 内容 | 状态 | 说明文档 |
|---|---|-----:|-----|
| Stage 0 | 项目初始化 | ✅ | [markdown/stage_0_project_init.md](markdown/stage_0_project_init.md) |
| Stage 1 | 字符级 Bigram 语言模型 | ✅ | [markdown/stage_1_bigram.md](markdown/stage_1_bigram.md) |
| Stage 2 | Embedding 语言模型 | ✅ | [markdown/stage_2_embedding_lm.md](markdown/stage_2_embedding_lm.md) |
| Stage 3 | Single-Head Causal Self-Attention | ✅ | [markdown/stage_3_single_head_attention.md](markdown/stage_3_single_head_attention.md) |
| Stage 4 | Multi-Head Causal Self-Attention | ✅ | [markdown/stage_4_multi_head_attention.md](markdown/stage_4_multi_head_attention.md) |
| Stage 5 | Transformer Block | ✅ | [markdown/stage_5_transformer_block.md](markdown/stage_5_transformer_block.md) |
| Stage 6 | 完整 Decoder-only GPT |  | [markdown/stage_6_decoder_only_gpt.md](markdown/stage_6_decoder_only_gpt.md) |
| Stage 7 | 生成策略优化 |  | [markdown/stage_7_generation_strategy.md](markdown/stage_7_generation_strategy.md) |
| Stage 8 | SFT 指令微调 |  | [markdown/stage_8_sft.md](markdown/stage_8_sft.md) |
| Stage 9 | LoRA 微调 |  | [markdown/stage_9_lora.md](markdown/stage_9_lora.md) |
| Stage 10 | RAG 知识库问答 |  | [markdown/stage_10_rag.md](markdown/stage_10_rag.md) |

历史阶段的训练、推理、配置和学习重点已迁移到上表对应的 Markdown 文件中。

## 当前阶段

当前阶段是 Stage 6：完整 Decoder-only GPT。

Stage 6 在 Stage 5 Transformer Block 的基础上，实现完整 Decoder-only GPT 主干，包括多层 Transformer Block 堆叠、final LayerNorm、`lm_head`、训练、生成、attention 可视化和模型参数统计。

当前阶段不实现：

- SFT
- LoRA
- RAG
- BPE tokenizer
- KV Cache
- Flash Attention
- 量化
- 模型服务部署
- `transformers`
- `datasets`
- `peft`
- `accelerate`
- `langchain`
- `llama-index`

## 部署环境

建议使用 Python 3.11：

```bash
conda create -n mini_gpt_edu python=3.11 -y
conda activate mini_gpt_edu
pip install -r requirements.txt
```

主要依赖保持轻量：

- `torch`
- `numpy`
- `matplotlib`
- `pyyaml`
- `tqdm`

目标设备：

- MacBook / Apple Silicon，使用 `device: auto` 自动选择 MPS 或 CPU
- RTX 4090 24GB，使用 `device: auto` 自动选择 CUDA
- CPU 调试

## 数据准备

Mac 小规模调试默认使用：

```text
data/raw.txt
```

RTX 4090 较大配置当前按 `configs/gpt_4090.yaml` 使用：

```text
data/computer_knowledge/raw/full_corpus.txt
```

如果本地数据路径不同，请先调整对应配置文件中的 `data.raw_text_path`。

## 最新训练和推理指令

以下命令对应当前最新阶段 Stage 6：完整 Decoder-only GPT。

Mac 快速测试：

```bash
python -m mini_gpt.train_gpt \
  --config configs/gpt_mac.yaml \
  --max-iters 20
```

Mac 小规模训练：

```bash
python -m mini_gpt.train_gpt \
  --config configs/gpt_mac.yaml
```

RTX 4090 24GB 较大配置训练：

```bash
python -m mini_gpt.train_gpt \
  --config configs/gpt_4090.yaml
```

Mac 配置 checkpoint 推理：

```bash
python -m mini_gpt.generate_gpt \
  --ckpt checkpoints/gpt_best.pt \
  --prompt "人工智能" \
  --temperature 1.0 \
  --top-k 50
```

4090 配置 checkpoint 推理：

```bash
python -m mini_gpt.generate_gpt \
  --ckpt checkpoints/gpt_4090_best.pt \
  --prompt "人工智能" \
  --temperature 0.9 \
  --top-k 50
```

GPT attention 可视化：

```bash
python -m mini_gpt.visualize_gpt_attention \
  --ckpt checkpoints/gpt_best.pt \
  --prompt "人工智能"
```

模型参数统计：

```bash
python -m mini_gpt.model_summary \
  --config configs/gpt_mac.yaml
```

## 其他项目文档

- [PROJECT_PLAN.md](PROJECT_PLAN.md)：项目路线图。
- [STAGE_TASKS.md](STAGE_TASKS.md)：当前阶段任务边界。
- [AGENTS.md](AGENTS.md)：面向 Codex 的项目协作约束。
