"""Train the Stage 2 character-level Embedding language model.

Stage 3 修改：改为复用 mini_gpt.training 中的通用训练函数，
原有 Stage 2 训练命令保持不变。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from mini_gpt.dataset import get_batch, read_text, train_val_split
from mini_gpt.embedding_lm import EmbeddingLanguageModel
from mini_gpt.tokenizer import CharTokenizer
from mini_gpt.training import estimate_loss, load_config
from mini_gpt.utils import get_device, save_checkpoint, set_seed


def main() -> None:
    """Run Embedding language model training from command line arguments."""
    parser = argparse.ArgumentParser(description="Train an Embedding language model.")
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

    set_seed(int(train_config["seed"]))

    device = get_device(train_config["device"])

    text = read_text(data_config["raw_text_path"])
    tokenizer = CharTokenizer.from_text(text)
    token_ids = tokenizer.encode(text)
    train_data, val_data = train_val_split(token_ids, data_config["train_ratio"])

    vocab_size = tokenizer.vocab_size
    block_size = int(model_config["block_size"])
    n_embd = int(model_config["n_embd"])
    batch_size = int(train_config["batch_size"])
    max_iters = int(args.max_iters or train_config["max_iters"])
    eval_interval = int(train_config["eval_interval"])
    eval_iters = int(train_config["eval_iters"])
    learning_rate = float(train_config["learning_rate"])

    print(f"Using device: {device}")
    print(f"vocab_size: {vocab_size}")
    print(f"block_size: {block_size}")
    print(f"n_embd: {n_embd}")
    print(f"batch_size: {batch_size}")
    print(f"max_iters: {max_iters}")

    if len(train_data) <= block_size:
        raise ValueError(
            f"训练数据长度 {len(train_data)} 必须大于 block_size {block_size}。"
            "请减小 block_size 或增加训练文本。"
        )

    model = EmbeddingLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
    ).to(device)
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
                tokenizer.save(tokenizer_path)
                save_checkpoint(
                    path=checkpoint_path,
                    model=model,
                    config=config,
                    tokenizer_path=tokenizer_path,
                    step=iter_index,
                    best_val_loss=best_val_loss,
                )

        # x shape: [batch_size, block_size]
        # y shape: [batch_size, block_size]
        x, y = get_batch(train_data, batch_size, block_size, device)

        _, loss = model(x, y)
        if loss is None:
            raise RuntimeError("loss 不应为空，请检查 targets 是否传入模型")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        progress.set_postfix(loss=f"{loss.item():.4f}")

    print(f"Best checkpoint saved to: {checkpoint_path}")
    print(f"Tokenizer saved to: {tokenizer_path}")


if __name__ == "__main__":
    main()
