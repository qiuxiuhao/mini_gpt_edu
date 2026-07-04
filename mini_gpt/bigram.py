"""A minimal character-level Bigram language model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    """Bigram model that predicts the next token from the current token.

    The model uses one embedding table. For each input token id, the embedding
    vector directly represents logits for the next token.
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the model and optionally compute cross entropy loss.

        idx shape: [batch_size, block_size]
        targets shape: [batch_size, block_size]
        logits shape: [batch_size, block_size, vocab_size]
        """
        logits = self.token_embedding_table(idx)

        loss = None
        if targets is not None:
            batch_size, block_size, vocab_size = logits.shape

            # logits_flat shape: [batch_size * block_size, vocab_size]
            logits_flat = logits.view(batch_size * block_size, vocab_size)

            # targets_flat shape: [batch_size * block_size]
            targets_flat = targets.view(batch_size * block_size)

            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Generate new token ids from a starting sequence.

        idx shape: [batch_size, current_length]
        output shape: [batch_size, current_length + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            logits, _ = self(idx)

            # last_logits shape: [batch_size, vocab_size]
            last_logits = logits[:, -1, :]

            # probs shape: [batch_size, vocab_size]
            probs = F.softmax(last_logits, dim=-1)

            # next_id shape: [batch_size, 1]
            next_id = torch.multinomial(probs, num_samples=1)

            # idx shape grows by one token each step.
            idx = torch.cat((idx, next_id), dim=1)

        return idx
