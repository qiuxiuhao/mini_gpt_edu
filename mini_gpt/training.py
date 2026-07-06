"""Stage 3 抽取：训练脚本通用工具。

Stage 1 Bigram、Stage 2 Embedding LM、Stage 3 Attention LM 都会用到
配置读取和 loss 评估。把这些逻辑抽出来后，原有训练命令仍然保持不变。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from mini_gpt.dataset import get_batch


def load_config(path: str | Path) -> dict[str, Any]:
    """Stage 3 抽取：统一读取 YAML 配置文件。"""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: torch.device,
) -> dict[str, float]:
    """Stage 3 抽取：统一估计训练集和验证集 loss。

    model(x, y) 至少返回:
    - logits shape: [batch_size, block_size, vocab_size]
    - loss: scalar tensor

    Stage 3 的 AttentionLanguageModel 在 return_attention=True 时可以返回
    attention weight；这里默认不请求 attention，因此仍兼容所有阶段模型。
    """
    model.eval()
    losses: dict[str, float] = {}

    for split_name, data in {"train": train_data, "val": val_data}.items():
        # The validation text can be tiny in this teaching project, so use a
        # smaller eval block if needed. Training still uses the configured
        # block_size.
        eval_block_size = min(block_size, len(data) - 1)
        if eval_block_size < 1:
            losses[split_name] = float("nan")
            continue

        split_losses = torch.zeros(eval_iters)
        for iter_index in range(eval_iters):
            x, y = get_batch(data, batch_size, eval_block_size, device)
            model_output = model(x, y)
            loss = model_output[1]
            if loss is None:
                raise RuntimeError("loss 不应为空，请检查 targets 是否传入模型")
            split_losses[iter_index] = loss.item()

        losses[split_name] = split_losses.mean().item()

    model.train()
    return losses
