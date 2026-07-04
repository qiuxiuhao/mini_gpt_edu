"""Generate text with the Stage 1 Bigram language model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mini_gpt.bigram import BigramLanguageModel
from mini_gpt.tokenizer import CharTokenizer


def get_device(device_name: str) -> torch.device:
    """Choose CPU or Apple Silicon MPS device."""
    if device_name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("参数要求使用 mps，但当前 PyTorch 环境不可用 mps")

    if device_name not in {"cpu", "mps"}:
        raise ValueError("device 只能是 auto、cpu 或 mps")

    return torch.device(device_name)


def main() -> None:
    """Run text generation from command line arguments."""
    parser = argparse.ArgumentParser(description="Generate text with a Bigram model.")
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
        choices=["auto", "cpu", "mps"],
        help="Device to run generation on.",
    )
    args = parser.parse_args()

    device = get_device(args.device)
    checkpoint_path = Path(args.ckpt)
    tokenizer_path = checkpoint_path.parent / "tokenizer.json"

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    tokenizer = CharTokenizer.load(tokenizer_path)

    model = BigramLanguageModel(vocab_size=checkpoint["vocab_size"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
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
