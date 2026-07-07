"""Stage 5 新增：a minimal pre-LN Transformer Block.

This module reuses the Stage 4 Multi-Head Causal Self-Attention and adds
LayerNorm, Residual Connection, and FeedForward. It does not implement a full
Decoder-only GPT, SFT, LoRA, or RAG.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mini_gpt.attention import MultiHeadCausalSelfAttention


class FeedForward(nn.Module):
    """Stage 5 新增：position-wise MLP inside a Transformer Block."""

    def __init__(self, n_embd: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Stage 5 新增：run the FeedForward network.

        x shape: [batch_size, block_size, n_embd]
        hidden shape: [batch_size, block_size, 4 * n_embd]
        output shape: [batch_size, block_size, n_embd]
        """
        return self.net(x)


class TransformerBlock(nn.Module):
    """Stage 5 新增：one pre-LN Transformer Block.

    Pre-LN structure:
    x = x + Attention(LayerNorm(x))
    x = x + FeedForward(LayerNorm(x))
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.block_size = block_size
        self.dropout = dropout

        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadCausalSelfAttention(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
            dropout=dropout,
        )
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd=n_embd, dropout=dropout)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Stage 5 新增：run one Transformer Block.

        x shape: [batch_size, block_size, n_embd]
        ln1_out shape: [batch_size, block_size, n_embd]
        attention_out shape: [batch_size, block_size, n_embd]
        after_attention_res shape: [batch_size, block_size, n_embd]
        ln2_out shape: [batch_size, block_size, n_embd]
        feed_forward_out shape: [batch_size, block_size, n_embd]
        block_out shape: [batch_size, block_size, n_embd]
        attention_weights shape: [batch_size, n_head, block_size, block_size]
        """
        # ln1_out shape: [batch_size, block_size, n_embd]
        ln1_out = self.ln1(x)

        # attention_out shape: [batch_size, block_size, n_embd]
        attention_out, attention_weights = self.attn(
            ln1_out,
            return_attention=return_attention,
        )

        # x shape: [batch_size, block_size, n_embd]
        x = x + attention_out

        # ln2_out shape: [batch_size, block_size, n_embd]
        ln2_out = self.ln2(x)

        # feed_forward_out shape: [batch_size, block_size, n_embd]
        feed_forward_out = self.ffn(ln2_out)

        # x shape: [batch_size, block_size, n_embd]
        x = x + feed_forward_out

        if return_attention:
            return x, attention_weights
        return x, None
