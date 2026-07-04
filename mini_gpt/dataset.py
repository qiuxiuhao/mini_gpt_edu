"""Dataset helpers for character-level next-token prediction."""

from __future__ import annotations

from pathlib import Path

import torch


def read_text(path: str | Path) -> str:
    """Read raw training text from a UTF-8 text file."""
    return Path(path).read_text(encoding="utf-8")


def train_val_split(
    token_ids: list[int], train_ratio: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split token ids into train and validation tensors.

    token_ids shape: [num_tokens]
    train_data shape: [num_train_tokens]
    val_data shape: [num_val_tokens]
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio 必须在 0 和 1 之间")
    if len(token_ids) < 2:
        raise ValueError("训练文本太短，至少需要 2 个 token")

    split_index = int(len(token_ids) * train_ratio)
    split_index = max(1, min(split_index, len(token_ids) - 1))

    train_data = torch.tensor(token_ids[:split_index], dtype=torch.long)
    val_data = torch.tensor(token_ids[split_index:], dtype=torch.long)
    return train_data, val_data


def get_batch(
    data: torch.Tensor, batch_size: int, block_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch for next-token prediction.

    x contains the current characters. y contains the next characters.

    x shape: [batch_size, block_size]
    y shape: [batch_size, block_size]
    """
    if len(data) <= block_size:
        raise ValueError(
            f"数据长度 {len(data)} 必须大于 block_size {block_size}，"
            "否则无法构造 next-token 样本"
        )

    start_positions = torch.randint(0, len(data) - block_size, (batch_size,))

    # x shape: [batch_size, block_size]
    x = torch.stack([data[i : i + block_size] for i in start_positions])

    # y shape: [batch_size, block_size]
    # y is x shifted one character to the right.
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in start_positions])

    return x.to(device), y.to(device)
