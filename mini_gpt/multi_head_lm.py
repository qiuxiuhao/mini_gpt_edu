"""Stage 4 新增：language model with Multi-Head Causal Self-Attention.

The model keeps the Stage 3 teaching flow, but replaces one attention head
with multiple causal heads, concat, and output projection. It does not
implement Transformer Block, LayerNorm, FeedForward, or Residual Connection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_gpt.attention import MultiHeadCausalSelfAttention


class MultiHeadLanguageModel(nn.Module):
    """Stage 4 新增：Embedding LM plus Multi-Head Causal Self-Attention."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_head: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.dropout = dropout

        if n_embd % n_head != 0:
            raise ValueError(
                f"n_embd {n_embd} 必须能被 n_head {n_head} 整除，"
                "这样才能计算 head_size = n_embd // n_head。"
            )
        self.head_size = n_embd // n_head

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.attention = MultiHeadCausalSelfAttention(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
            dropout=dropout,
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor | None
    ]:
        """Stage 4 新增：run Multi-Head LM and optionally return attention.

        idx shape: [batch_size, block_size]
        targets shape: [batch_size, block_size]
        logits shape: [batch_size, block_size, vocab_size]
        attention_weights shape: [batch_size, n_head, block_size, block_size]
        """
        batch_size, block_size = idx.shape
        if block_size > self.block_size:
            raise ValueError(
                f"输入长度 {block_size} 不能超过 block_size {self.block_size}"
            )

        # token_emb shape: [batch_size, block_size, n_embd]
        token_emb = self.token_embedding_table(idx)

        # pos_ids shape: [block_size]
        pos_ids = torch.arange(block_size, device=idx.device)

        # pos_emb shape: [block_size, n_embd]
        pos_emb = self.position_embedding_table(pos_ids)

        # x shape: [batch_size, block_size, n_embd]
        x = token_emb + pos_emb

        # x shape after multi-head attention: [batch_size, block_size, n_embd]
        x, attention_weights = self.attention(
            x,
            return_attention=return_attention,
        )

        # logits shape: [batch_size, block_size, vocab_size]
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            batch_size, block_size, vocab_size = logits.shape

            # logits_flat shape: [batch_size * block_size, vocab_size]
            logits_flat = logits.view(batch_size * block_size, vocab_size)

            # targets_flat shape: [batch_size * block_size]
            targets_flat = targets.view(batch_size * block_size)

            loss = F.cross_entropy(logits_flat, targets_flat)

        if return_attention:
            return logits, loss, attention_weights
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Stage 4 新增：generate token ids autoregressively.

        idx shape: [batch_size, current_length]
        output shape: [batch_size, current_length + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # idx_cond shape: [batch_size, <= block_size]
            idx_cond = idx[:, -self.block_size :]

            logits, _ = self(idx_cond)

            # last_logits shape: [batch_size, vocab_size]
            last_logits = logits[:, -1, :]

            # probs shape: [batch_size, vocab_size]
            probs = F.softmax(last_logits, dim=-1)

            # next_id shape: [batch_size, 1]
            next_id = torch.multinomial(probs, num_samples=1)

            # idx shape grows by one token each step.
            idx = torch.cat((idx, next_id), dim=1)

        return idx
