"""Train the Stage 1 character-level Bigram language model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from mini_gpt.bigram import BigramLanguageModel
from mini_gpt.dataset import get_batch, read_text, train_val_split
from mini_gpt.tokenizer import CharTokenizer


def load_config(path: str | Path) -> dict:
    """Load YAML config from disk."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def get_device(device_name: str) -> torch.device:
    """Choose CPU or Apple Silicon MPS device."""
    if device_name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("配置要求使用 mps，但当前 PyTorch 环境不可用 mps")

    if device_name not in {"cpu", "mps"}:
        raise ValueError("device 只能是 auto、cpu 或 mps")

    return torch.device(device_name)


@torch.no_grad()
def estimate_loss(
    model: BigramLanguageModel,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: torch.device,
) -> dict[str, float]:
    """Estimate train and validation loss using random batches."""
    model.eval()
    losses: dict[str, float] = {}

    for split_name, data in {"train": train_data, "val": val_data}.items():
        # The validation text is tiny in this teaching project, so use a
        # smaller eval block if needed. Training still uses the configured
        # block_size.
        eval_block_size = min(block_size, len(data) - 1)
        if eval_block_size < 1:
            losses[split_name] = float("nan")
            continue

        split_losses = torch.zeros(eval_iters)
        for iter_index in range(eval_iters):
            x, y = get_batch(data, batch_size, eval_block_size, device)
            _, loss = model(x, y)
            split_losses[iter_index] = loss.item()

        losses[split_name] = split_losses.mean().item()

    model.train()
    return losses


def save_checkpoint(
    path: str | Path,
    model: BigramLanguageModel,
    vocab_size: int,
    block_size: int,
    config: dict,
    best_val_loss: float,
) -> None:
    """Save the best model checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": vocab_size,
            "block_size": block_size,
            "config": config,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def main() -> None:
    """Run Bigram training from command line arguments."""
    parser = argparse.ArgumentParser(description="Train a character Bigram model.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--max-iters",
        type=int,
        default=None,
        help="Override train.max_iters for quick tests.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config["data"]
    model_config = config["model"]
    train_config = config["train"]
    output_config = config["output"]

    torch.manual_seed(train_config["seed"])

    device = get_device(train_config["device"])
    print(f"Using device: {device}")

    text = read_text(data_config["raw_text_path"])
    tokenizer = CharTokenizer.from_text(text)
    token_ids = tokenizer.encode(text)
    train_data, val_data = train_val_split(token_ids, data_config["train_ratio"])

    block_size = int(model_config["block_size"])
    batch_size = int(train_config["batch_size"])
    max_iters = int(args.max_iters or train_config["max_iters"])
    eval_interval = int(train_config["eval_interval"])
    eval_iters = int(train_config["eval_iters"])
    learning_rate = float(train_config["learning_rate"])

    if len(train_data) <= block_size:
        raise ValueError(
            f"训练数据长度 {len(train_data)} 必须大于 block_size {block_size}。"
            "请减小 block_size 或增加 data/raw.txt 文本。"
        )

    model = BigramLanguageModel(vocab_size=tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    checkpoint_dir = Path(output_config["checkpoint_dir"])
    checkpoint_path = checkpoint_dir / output_config["best_ckpt_name"]
    tokenizer_path = checkpoint_dir / output_config["tokenizer_name"]

    best_val_loss = float("inf")
    progress = tqdm(range(max_iters), desc="Training")

    for iter_index in progress:
        if iter_index % eval_interval == 0 or iter_index == max_iters - 1:
            losses = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                batch_size=batch_size,
                block_size=block_size,
                eval_iters=eval_iters,
                device=device,
            )
            train_loss = losses["train"]
            val_loss = losses["val"]
            print(
                f"step {iter_index}: train loss {train_loss:.4f}, "
                f"val loss {val_loss:.4f}"
            )

            loss_for_best = val_loss if not torch.isnan(torch.tensor(val_loss)) else train_loss
            if loss_for_best < best_val_loss:
                best_val_loss = loss_for_best
                save_checkpoint(
                    path=checkpoint_path,
                    model=model,
                    vocab_size=tokenizer.vocab_size,
                    block_size=block_size,
                    config=config,
                    best_val_loss=best_val_loss,
                )
                tokenizer.save(tokenizer_path)

        # x shape: [batch_size, block_size]
        # y shape: [batch_size, block_size]
        x, y = get_batch(train_data, batch_size, block_size, device)

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        progress.set_postfix(loss=f"{loss.item():.4f}")

    print(f"Best checkpoint saved to: {checkpoint_path}")
    print(f"Tokenizer saved to: {tokenizer_path}")


if __name__ == "__main__":
    main()
