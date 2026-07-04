"""Character-level tokenizer for Stage 1.

This module turns raw text into integer token ids, and turns token ids back
into text. In Stage 1, one character is one token.
"""

from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:
    """A small character-level tokenizer.

    The tokenizer stores two mappings:
    - stoi: string to integer id
    - itos: integer id to string
    """

    def __init__(self, stoi: dict[str, int], itos: list[str]) -> None:
        self.stoi = stoi
        self.itos = itos

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Build a tokenizer from all unique characters in text."""
        chars = sorted(set(text))
        stoi = {char: index for index, char in enumerate(chars)}
        itos = chars
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        """Return the number of unique characters."""
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        """Convert text into token ids.

        Example:
        text: "人工智能"
        token_ids shape: [text_length]
        """
        unknown_chars = sorted({char for char in text if char not in self.stoi})
        if unknown_chars:
            raise ValueError(f"文本中包含词表外字符: {unknown_chars}")

        return [self.stoi[char] for char in text]

    def decode(self, ids: list[int]) -> str:
        """Convert token ids back into text.

        ids shape: [text_length]
        """
        chars: list[str] = []
        for token_id in ids:
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(f"token id 超出词表范围: {token_id}")
            chars.append(self.itos[token_id])
        return "".join(chars)

    def save(self, path: str | Path) -> None:
        """Save tokenizer mappings as a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "stoi": self.stoi,
            "itos": self.itos,
            "vocab_size": self.vocab_size,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        """Load a tokenizer from a JSON file."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))

        itos = list(data["itos"])
        stoi = {char: int(index) for char, index in data["stoi"].items()}
        return cls(stoi=stoi, itos=itos)
