"""Stage 6 新增：generate text with the complete Decoder-only GPT."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mini_gpt.gpt import GPTLanguageModel
from mini_gpt.tokenizer import CharTokenizer
from mini_gpt.utils import get_device, load_checkpoint, resolve_tokenizer_path


def main() -> None:
    """Stage 6 新增：run GPT text generation from command line arguments."""
    parser = argparse.ArgumentParser(description="Generate text with a GPT model.")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint.")
    parser.add_argument("--prompt", required=True, help="Prompt text.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Number of new characters to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Scale logits before sampling. Must be > 0.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Keep only the top k logits before sampling.",
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
    generate_config = config.get("generate", {})

    tokenizer_path = resolve_tokenizer_path(
        checkpoint_path=checkpoint_path,
        tokenizer_path=checkpoint["tokenizer_path"],
    )
    tokenizer = CharTokenizer.load(tokenizer_path)

    vocab_size = int(checkpoint["vocab_size"])
    block_size = int(model_config["block_size"])
    n_embd = int(model_config["n_embd"])
    n_head = int(model_config["n_head"])
    n_layer = int(model_config.get("n_layer", checkpoint.get("n_layer", 1)))
    dropout = float(model_config.get("dropout", 0.0))
    max_new_tokens = int(
        args.max_new_tokens
        if args.max_new_tokens is not None
        else generate_config.get("max_new_tokens", 100)
    )
    temperature = float(
        args.temperature
        if args.temperature is not None
        else generate_config.get("temperature", 1.0)
    )
    top_k = args.top_k if args.top_k is not None else generate_config.get("top_k")
    if top_k is not None:
        top_k = int(top_k)

    model = GPTLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt)

    # idx shape: [batch_size=1, prompt_length]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    generated_ids = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    # generated_ids[0] shape: [prompt_length + max_new_tokens]
    output_text = tokenizer.decode(generated_ids[0].tolist())
    print(output_text)


if __name__ == "__main__":
    main()
