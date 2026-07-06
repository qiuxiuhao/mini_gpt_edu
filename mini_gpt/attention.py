"""Stage 3 新增：single-head causal self-attention language model.

This module adds one attention head after token embedding + position embedding.
It is intentionally small and does not implement multi-head attention,
Transformer blocks, LayerNorm, FFN, LoRA, SFT, or RAG.

Stage 4 修改：追加 Multi-Head Causal Self-Attention 教学实现。
Stage 4 只增加多个 causal attention head、concat 和 output projection，
不实现 Transformer Block、LayerNorm、FeedForward 或 Residual Connection。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHeadCausalSelfAttention(nn.Module):
    """Stage 3 新增：a single causal self-attention head.

    The module teaches the full attention path:
    Q, K, V -> QK^T / sqrt(d) -> causal mask -> softmax -> weight @ V.
    """

    def __init__(self, n_embd: int, head_size: int, block_size: int) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.head_size = head_size
        self.block_size = block_size

        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # causal_mask shape: [block_size, block_size]
        # True means "this token is allowed to attend to that token".
        causal_mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Stage 3 新增：run one attention head.

        x shape: [batch_size, sequence_length, n_embd]
        attention_out shape: [batch_size, sequence_length, head_size]
        attention_weight shape: [batch_size, sequence_length, sequence_length]
        """
        _, sequence_length, _ = x.shape

        # q, k, v shape: [batch_size, sequence_length, head_size]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # attention_score shape: [batch_size, sequence_length, sequence_length]
        attention_score = q @ k.transpose(-2, -1)
        attention_score = attention_score / math.sqrt(self.head_size)

        # mask shape: [sequence_length, sequence_length]
        mask = self.causal_mask[:sequence_length, :sequence_length]

        # Future positions are set to -inf before softmax.
        attention_score = attention_score.masked_fill(~mask, float("-inf"))

        # attention_weight shape: [batch_size, sequence_length, sequence_length]
        attention_weight = F.softmax(attention_score, dim=-1)

        # attention_out shape: [batch_size, sequence_length, head_size]
        attention_out = attention_weight @ v

        if return_attention:
            return attention_out, attention_weight
        return attention_out, None


class AttentionLanguageModel(nn.Module):
    """Stage 3 新增：Embedding LM plus one causal self-attention head."""

    def __init__(
        self, vocab_size: int, block_size: int, n_embd: int, head_size: int
    ) -> None:
        super().__init__()
        if head_size != n_embd:
            raise ValueError(
                "Stage 3 教学版要求 head_size == n_embd，"
                "避免提前引入 output projection。"
            )

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.head_size = head_size

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.attention = SingleHeadCausalSelfAttention(
            n_embd=n_embd,
            head_size=head_size,
            block_size=block_size,
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
        """Stage 3 新增：run Attention LM and optionally return attention weights.

        idx shape: [batch_size, sequence_length]
        targets shape: [batch_size, sequence_length]
        logits shape: [batch_size, sequence_length, vocab_size]
        attention_weight shape: [batch_size, sequence_length, sequence_length]
        """
        batch_size, sequence_length = idx.shape
        if sequence_length > self.block_size:
            raise ValueError(
                f"输入长度 {sequence_length} 不能超过 block_size {self.block_size}"
            )

        # token_emb shape: [batch_size, sequence_length, n_embd]
        token_emb = self.token_embedding_table(idx)

        # pos_ids shape: [sequence_length]
        pos_ids = torch.arange(sequence_length, device=idx.device)

        # pos_emb shape: [sequence_length, n_embd]
        pos_emb = self.position_embedding_table(pos_ids)

        # x shape: [batch_size, sequence_length, n_embd]
        x = token_emb + pos_emb

        # x shape after attention: [batch_size, sequence_length, n_embd]
        x, attention_weight = self.attention(x, return_attention=return_attention)

        # logits shape: [batch_size, sequence_length, vocab_size]
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            batch_size, sequence_length, vocab_size = logits.shape

            # logits_flat shape: [batch_size * sequence_length, vocab_size]
            logits_flat = logits.view(batch_size * sequence_length, vocab_size)

            # targets_flat shape: [batch_size * sequence_length]
            targets_flat = targets.view(batch_size * sequence_length)

            loss = F.cross_entropy(logits_flat, targets_flat)

        if return_attention:
            return logits, loss, attention_weight
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Stage 3 新增：generate token ids autoregressively.

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


class CausalSelfAttentionHead(nn.Module):
    """Stage 4 新增：one head used inside Multi-Head Causal Self-Attention.

    This class is intentionally similar to the Stage 3 single head, but is
    named for Stage 4 so students can clearly see each head in a multi-head
    module has its own Q, K, and V projections.
    """

    def __init__(self, n_embd: int, head_size: int, block_size: int) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.head_size = head_size
        self.block_size = block_size

        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # causal_mask shape: [block_size, block_size]
        causal_mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Stage 4 新增：run causal self-attention for one head.

        x shape: [batch_size, block_size, n_embd]
        head_out shape: [batch_size, block_size, head_size]
        attention_weight shape: [batch_size, block_size, block_size]
        """
        _, block_size, _ = x.shape

        # q shape: [batch_size, block_size, head_size]
        q = self.query(x)

        # k shape: [batch_size, block_size, head_size]
        k = self.key(x)

        # v shape: [batch_size, block_size, head_size]
        v = self.value(x)

        # attention_score shape: [batch_size, block_size, block_size]
        attention_score = q @ k.transpose(-2, -1)
        attention_score = attention_score / math.sqrt(self.head_size)

        # mask shape: [block_size, block_size]
        mask = self.causal_mask[:block_size, :block_size]
        attention_score = attention_score.masked_fill(~mask, float("-inf"))

        # attention_weight shape: [batch_size, block_size, block_size]
        attention_weight = F.softmax(attention_score, dim=-1)

        # head_out shape: [batch_size, block_size, head_size]
        head_out = attention_weight @ v

        if return_attention:
            return head_out, attention_weight
        return head_out, None


class MultiHeadCausalSelfAttention(nn.Module):
    """Stage 4 新增：multiple causal attention heads plus output projection."""

    def __init__(
        self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(
                f"n_embd {n_embd} 必须能被 n_head {n_head} 整除，"
                "这样才能计算 head_size = n_embd // n_head。"
            )

        self.n_embd = n_embd
        self.n_head = n_head
        self.block_size = block_size
        self.dropout = dropout
        self.head_size = n_embd // n_head

        self.heads = nn.ModuleList(
            [
                CausalSelfAttentionHead(
                    n_embd=n_embd,
                    head_size=self.head_size,
                    block_size=block_size,
                )
                for _ in range(n_head)
            ]
        )
        self.output_projection = nn.Linear(n_embd, n_embd)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Stage 4 新增：run all heads, concat outputs, then project.

        x shape: [batch_size, block_size, n_embd]
        concat_out shape: [batch_size, block_size, n_embd]
        out shape: [batch_size, block_size, n_embd]
        attention_weights shape: [batch_size, n_head, block_size, block_size]
        """
        head_outputs: list[torch.Tensor] = []
        attention_weights: list[torch.Tensor] = []

        for head in self.heads:
            # head_out shape: [batch_size, block_size, head_size]
            head_out, attention_weight = head(x, return_attention=return_attention)
            head_outputs.append(head_out)

            if return_attention:
                if attention_weight is None:
                    raise RuntimeError("attention_weight 不应为空")
                attention_weights.append(attention_weight)

        # concat_out shape: [batch_size, block_size, n_embd]
        concat_out = torch.cat(head_outputs, dim=-1)

        # out shape: [batch_size, block_size, n_embd]
        out = self.output_projection(concat_out)
        out = self.proj_dropout(out)

        if return_attention:
            # attention_weights shape: [batch_size, n_head, block_size, block_size]
            stacked_attention = torch.stack(attention_weights, dim=1)
            return out, stacked_attention
        return out, None
