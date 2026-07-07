"""Stage 6 新增：print GPT model structure and parameter counts."""

from __future__ import annotations

import argparse

import torch

from mini_gpt.dataset import read_text
from mini_gpt.gpt import GPTLanguageModel
from mini_gpt.tokenizer import CharTokenizer
from mini_gpt.training import load_config


def count_parameters(module: torch.nn.Module, trainable_only: bool = False) -> int:
    """Stage 6 新增：count parameters in a module."""
    parameters = module.parameters()
    if trainable_only:
        return sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def main() -> None:
    """Stage 6 新增：create a GPT model from config and print parameter counts."""
    parser = argparse.ArgumentParser(description="Print GPT model summary.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config["data"]
    model_config = config["model"]

    text = read_text(data_config["raw_text_path"])
    tokenizer = CharTokenizer.from_text(text)
    vocab_size = tokenizer.vocab_size

    block_size = int(model_config["block_size"])
    n_embd = int(model_config["n_embd"])
    n_head = int(model_config["n_head"])
    n_layer = int(model_config["n_layer"])
    dropout = float(model_config.get("dropout", 0.0))

    model = GPTLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
    )

    print(model)
    print()
    print(f"vocab_size: {vocab_size}")
    print(f"block_size: {block_size}")
    print(f"n_embd: {n_embd}")
    print(f"n_head: {n_head}")
    print(f"head_size: {model.head_size}")
    print(f"n_layer: {n_layer}")
    print(f"dropout: {dropout}")
    print()
    print(f"total_parameters: {count_parameters(model):,}")
    print(f"trainable_parameters: {count_parameters(model, trainable_only=True):,}")
    print(
        "token_embedding_parameters: "
        f"{count_parameters(model.token_embedding_table):,}"
    )
    print(
        "position_embedding_parameters: "
        f"{count_parameters(model.position_embedding_table):,}"
    )
    print(f"transformer_blocks_parameters: {count_parameters(model.blocks):,}")
    print(f"final_layer_norm_parameters: {count_parameters(model.ln_f):,}")
    print(f"lm_head_parameters: {count_parameters(model.lm_head):,}")


if __name__ == "__main__":
    main()
