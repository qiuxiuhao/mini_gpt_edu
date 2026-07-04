"""Stage 2 Embedding language model.

This module teaches the first GPT-like input step:
token ids become token vectors, position ids become position vectors, and the
sum is projected back to vocabulary logits with lm_head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingLanguageModel(nn.Module):
    """A character-level language model with token and position embeddings.

    Compared with Stage 1 Bigram, token ids are no longer used directly as
    logits. They first become learnable vectors of size n_embd.
    """

    def __init__(self, vocab_size: int, block_size: int, n_embd: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the model and optionally compute cross entropy loss.

        idx shape: [batch_size, block_size]
        targets shape: [batch_size, block_size]
        logits shape: [batch_size, block_size, vocab_size]
        """
        batch_size, sequence_length = idx.shape
        if sequence_length > self.block_size:
            raise ValueError(
                f"输入长度 {sequence_length} 不能超过 block_size {self.block_size}"
            )

        # token_emb shape: [batch_size, block_size, n_embd]
        token_emb = self.token_embedding_table(idx)

        # pos_ids shape: [block_size]
        pos_ids = torch.arange(sequence_length, device=idx.device)

        # pos_emb shape: [block_size, n_embd]
        pos_emb = self.position_embedding_table(pos_ids)

        # x shape: [batch_size, block_size, n_embd]
        x = token_emb + pos_emb

        # logits shape: [batch_size, block_size, vocab_size]
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            batch_size, sequence_length, vocab_size = logits.shape

            # logits_flat shape: [batch_size * block_size, vocab_size]
            logits_flat = logits.view(batch_size * sequence_length, vocab_size)

            # targets_flat shape: [batch_size * block_size]
            targets_flat = targets.view(batch_size * sequence_length)

            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Generate token ids autoregressively.

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
