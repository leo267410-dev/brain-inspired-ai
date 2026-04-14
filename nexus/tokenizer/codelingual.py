"""CodeLingual tokenizer: a unified tokenizer for code and natural language."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class CodeLingualTokenizer:
    """
    A BPE-style tokenizer that handles both code and natural language.
    Supports special tokens for code structure (indent, dedent, newline).

    This is a simplified implementation; in production, you would train
    a full BPE tokenizer on a mixed code/text corpus.
    """

    # Special tokens
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    CODE_TOKEN = "<code>"
    TEXT_TOKEN = "<text>"
    THOUGHT_TOKEN = "<thought>"
    INDENT_TOKEN = "<indent>"
    DEDENT_TOKEN = "<dedent>"
    NEWLINE_TOKEN = "<newline>"

    SPECIAL_TOKENS = [
        PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN,
        CODE_TOKEN, TEXT_TOKEN, THOUGHT_TOKEN,
        INDENT_TOKEN, DEDENT_TOKEN, NEWLINE_TOKEN,
    ]

    def __init__(self, vocab_size: int = 48000):
        self.vocab_size = vocab_size

        # Initialize with special tokens + basic character vocabulary
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        # Add special tokens
        for i, token in enumerate(self.SPECIAL_TOKENS):
            self.token_to_id[token] = i
            self.id_to_token[i] = token

        # Add basic ASCII characters
        offset = len(self.SPECIAL_TOKENS)
        for i in range(32, 127):  # printable ASCII
            char = chr(i)
            self.token_to_id[char] = offset + i - 32
            self.id_to_token[offset + i - 32] = char

        self._next_id = offset + 95  # 127 - 32

        # BPE merges (would be learned from data in practice)
        self.merges: List[Tuple[str, str]] = []

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.PAD_TOKEN]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id[self.UNK_TOKEN]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id[self.BOS_TOKEN]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id[self.EOS_TOKEN]

    @property
    def code_token_id(self) -> int:
        return self.token_to_id[self.CODE_TOKEN]

    @property
    def text_token_id(self) -> int:
        return self.token_to_id[self.TEXT_TOKEN]

    def _add_token(self, token: str) -> int:
        """Add a new token to the vocabulary."""
        if token in self.token_to_id:
            return self.token_to_id[token]
        if self._next_id >= self.vocab_size:
            return self.unk_token_id
        tid = self._next_id
        self.token_to_id[token] = tid
        self.id_to_token[tid] = token
        self._next_id += 1
        return tid

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: input text string.
            add_special_tokens: whether to add BOS/EOS tokens.
            max_length: optional max sequence length.

        Returns:
            List of token IDs.
        """
        tokens = []
        if add_special_tokens:
            tokens.append(self.bos_token_id)

        # Simple character-level tokenization (fallback)
        for char in text:
            if char in self.token_to_id:
                tokens.append(self.token_to_id[char])
            else:
                tokens.append(self.unk_token_id)

        if add_special_tokens:
            tokens.append(self.eos_token_id)

        if max_length is not None:
            tokens = tokens[:max_length]

        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: list of token IDs.
            skip_special_tokens: whether to skip special tokens.

        Returns:
            Decoded string.
        """
        tokens = []
        special_ids = set(
            self.token_to_id[t] for t in self.SPECIAL_TOKENS
        )

        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            if tid in self.id_to_token:
                tokens.append(self.id_to_token[tid])
            else:
                tokens.append(self.UNK_TOKEN)

        return "".join(tokens)

    def encode_code(self, code: str) -> List[int]:
        """
        Encode code with structure-aware tokens.

        Args:
            code: source code string.

        Returns:
            List of token IDs with structure tokens.
        """
        tokens = [self.bos_token_id, self.code_token_id]
        lines = code.split("\n")

        prev_indent = 0
        for line in lines:
            # Detect indentation
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            indent_level = indent // 4  # Assume 4-space indentation

            # Add indent/dedent tokens
            if indent_level > prev_indent:
                for _ in range(indent_level - prev_indent):
                    tokens.append(self.token_to_id.get(self.INDENT_TOKEN, self.unk_token_id))
            elif indent_level < prev_indent:
                for _ in range(prev_indent - indent_level):
                    tokens.append(self.token_to_id.get(self.DEDENT_TOKEN, self.unk_token_id))

            # Encode the line content
            for char in stripped:
                if char in self.token_to_id:
                    tokens.append(self.token_to_id[char])
                else:
                    tokens.append(self.unk_token_id)

            tokens.append(self.token_to_id.get(self.NEWLINE_TOKEN, self.unk_token_id))
            prev_indent = indent_level

        tokens.append(self.eos_token_id)
        return tokens

    def save(self, path: str) -> None:
        """Save tokenizer to file."""
        data = {
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
            "merges": self.merges,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> "CodeLingualTokenizer":
        """Load tokenizer from file."""
        data = json.loads(Path(path).read_text())
        tokenizer = cls(vocab_size=data["vocab_size"])
        tokenizer.token_to_id = data["token_to_id"]
        tokenizer.id_to_token = {int(v): k for k, v in data["token_to_id"].items()}
        tokenizer.merges = [tuple(m) for m in data.get("merges", [])]
        return tokenizer
