"""Small shared utilities for MiniGPT-edu training and generation scripts.

Stage 3 修改：增加 Attention LM 可复用的 checkpoint/tokenizer 工具，
原有 Stage 1 和 Stage 2 命令保持不变。

Stage 4 修改：checkpoint 额外保存 Multi-Head LM 的 n_head。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds so experiments are easier to reproduce."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_name: str = "auto") -> torch.device:
    """Choose cuda, mps, or cpu for training and generation.

    auto priority:
    1. NVIDIA CUDA GPU
    2. Apple Silicon MPS
    3. CPU
    """
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("配置要求使用 cuda，但当前 PyTorch 环境不可用 cuda")

    if device_name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("配置要求使用 mps，但当前 PyTorch 环境不可用 mps")

    if device_name not in {"cpu", "mps", "cuda"}:
        raise ValueError("device 只能是 auto、cpu、mps 或 cuda")

    return torch.device(device_name)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    config: dict[str, Any],
    tokenizer_path: str | Path,
    step: int,
    best_val_loss: float,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """Save a model checkpoint with enough metadata to recreate the model."""
    path = Path(path)
    ensure_dir(path.parent)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "tokenizer_path": str(tokenizer_path),
        "step": step,
        "best_val_loss": best_val_loss,
        "vocab_size": getattr(model, "vocab_size", None),
        "block_size": getattr(model, "block_size", None),
        "n_embd": getattr(model, "n_embd", None),
        # Stage 3 修改：AttentionLanguageModel exposes head_size.
        "head_size": getattr(model, "head_size", None),
        # Stage 4 修改：MultiHeadLanguageModel exposes n_head.
        "n_head": getattr(model, "n_head", None),
    }
    if extra_metadata is not None:
        checkpoint.update(extra_metadata)

    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    """Load a checkpoint onto the requested device."""
    return torch.load(Path(path), map_location=device)


def resolve_tokenizer_path(
    checkpoint_path: str | Path, tokenizer_path: str | Path | None = None
) -> Path:
    """Stage 3 新增：find the tokenizer saved with a checkpoint.

    Stage 2 and Stage 3 checkpoints store tokenizer_path. Older Stage 1
    checkpoints may not, so this helper also falls back to tokenizer.json
    beside the checkpoint.
    """
    checkpoint_path = Path(checkpoint_path)

    if tokenizer_path is not None:
        path = Path(tokenizer_path)
        if path.exists():
            return path

        fallback_path = checkpoint_path.parent / path.name
        if fallback_path.exists():
            return fallback_path

    default_path = checkpoint_path.parent / "tokenizer.json"
    if default_path.exists():
        return default_path

    raise FileNotFoundError(f"找不到 tokenizer 文件: {tokenizer_path}")
