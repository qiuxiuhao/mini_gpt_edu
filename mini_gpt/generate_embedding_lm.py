"""Generate text with the Stage 2 Embedding language model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mini_gpt.embedding_lm import EmbeddingLanguageModel
from mini_gpt.tokenizer import CharTokenizer
from mini_gpt.utils import get_device, load_checkpoint


def resolve_tokenizer_path(checkpoint_path: Path, tokenizer_path: str) -> Path:
    """Find the tokenizer saved with a checkpoint."""
    path = Path(tokenizer_path)
    if path.exists():
        return path

    fallback_path = checkpoint_path.parent / path.name
    if fallback_path.exists():
        return fallback_path

    raise FileNotFoundError(f"找不到 tokenizer 文件: {tokenizer_path}")


def main() -> None:
    """Run text generation from command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate text with an Embedding language model."
    )
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint.")
    parser.add_argument("--prompt", required=True, help="Prompt text.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Number of new characters to generate.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Device to run generation on.",
    )
    args = parser.parse_args()

    device = get_device(args.device)
    checkpoint_path = Path(args.ckpt)
    checkpoint = load_checkpoint(checkpoint_path, device)

    config = checkpoint["config"]
    model_config = config["model"]

    tokenizer_path = resolve_tokenizer_path(
        checkpoint_path=checkpoint_path,
        tokenizer_path=checkpoint["tokenizer_path"],
    )
    tokenizer = CharTokenizer.load(tokenizer_path)

    vocab_size = int(checkpoint["vocab_size"])
    block_size = int(model_config["block_size"])
    n_embd = int(model_config["n_embd"])

    model = EmbeddingLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt)

    # idx shape: [batch_size=1, prompt_length]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    generated_ids = model.generate(idx, max_new_tokens=args.max_new_tokens)

    # generated_ids[0] shape: [prompt_length + max_new_tokens]
    output_text = tokenizer.decode(generated_ids[0].tolist())
    print(output_text)


if __name__ == "__main__":
    main()
