"""Stage 6 新增：visualize GPT attention weights by layer and head."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mini_gpt_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from mini_gpt.gpt import GPTLanguageModel
from mini_gpt.tokenizer import CharTokenizer
from mini_gpt.utils import get_device, load_checkpoint, resolve_tokenizer_path


def save_attention_heatmap(
    attention_weight: torch.Tensor,
    token_labels: list[str],
    layer_index: int,
    head_index: int,
    output_dir: Path,
) -> Path:
    """Stage 6 新增：save one GPT layer/head attention heatmap."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"layer_{layer_index}_head_{head_index}_attention.png"

    # matrix shape: [block_size, block_size]
    matrix = attention_weight.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(f"Layer {layer_index} Head {head_index} Attention")
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    ax.set_xticks(range(len(token_labels)))
    ax.set_yticks(range(len(token_labels)))
    ax.set_xticklabels(token_labels, rotation=45, ha="right")
    ax.set_yticklabels(token_labels)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return output_path


def main() -> None:
    """Stage 6 新增：load a GPT checkpoint and save attention heatmaps."""
    parser = argparse.ArgumentParser(description="Visualize GPT attention weights.")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint.")
    parser.add_argument("--prompt", required=True, help="Prompt text.")
    parser.add_argument(
        "--output-dir",
        default="outputs/gpt_attention",
        help="Directory to save attention heatmaps.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Device to run visualization on.",
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
    n_head = int(model_config["n_head"])
    n_layer = int(model_config.get("n_layer", checkpoint.get("n_layer", 1)))
    dropout = float(model_config.get("dropout", 0.0))

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
    prompt_ids = prompt_ids[-block_size:]
    token_text = tokenizer.decode(prompt_ids)
    token_labels = [str(index) for index in range(len(prompt_ids))]

    # idx shape: [batch_size=1, prompt_length]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        _, _, all_attention_weights = model(idx, return_attention=True)

    output_dir = Path(args.output_dir)
    saved_paths: list[Path] = []
    for layer_index, layer_attention in enumerate(all_attention_weights):
        # layer_attention shape: [batch_size=1, n_head, block_size, block_size]
        layer_attention = layer_attention[0]
        for head_index in range(layer_attention.shape[0]):
            saved_path = save_attention_heatmap(
                attention_weight=layer_attention[head_index],
                token_labels=token_labels,
                layer_index=layer_index,
                head_index=head_index,
                output_dir=output_dir,
            )
            saved_paths.append(saved_path)

    print(f"num_layers: {len(all_attention_weights)}")
    print(f"tokens: {token_text}")
    for saved_path in saved_paths:
        print(f"saved: {saved_path}")


if __name__ == "__main__":
    main()
