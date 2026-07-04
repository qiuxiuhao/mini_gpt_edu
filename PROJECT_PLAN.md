# 2. `PROJECT_PLAN.md`：整个项目路线图

这个是让 Codex 永远知道：我们不是随便写代码，而是在一步步做一个教学版 MiniGPT。

```markdown
# PROJECT_PLAN.md

# MiniGPT-edu 项目路线图

本项目目标是从零实现一个教学版 GPT 语言模型，帮助学习者从项目实战出发理解大模型核心算法。

## Stage 0：项目初始化

目标：

- 创建项目结构
- 配置 AGENTS.md
- 配置 requirements.txt
- 配置 README.md
- 准备最小 raw.txt 数据
- 准备基础配置文件

产出：

- 可被 Codex 稳定理解的项目骨架

---

## Stage 1：字符级 Bigram 语言模型

目标：

- 实现字符级 tokenizer
- 构建字符表 vocab
- 实现 encode / decode
- 构造 next-token prediction 数据
- 实现 BigramLanguageModel
- 实现训练脚本
- 实现生成脚本

重点理解：

- token
- vocab
- token id
- next-token prediction
- logits
- cross entropy
- generation

验收命令：

```bash
python -m mini_gpt.train_bigram --config configs/bigram.yaml
python -m mini_gpt.generate_bigram --ckpt checkpoints/bigram_best.pt --prompt "人工智能"