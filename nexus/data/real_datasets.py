"""Real dataset loaders for NEXUS-Ω training using HuggingFace datasets."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch.utils.data import Dataset


class WikiTextDataset(Dataset):
    """
    WikiText-2 dataset for language modeling.
    Downloads from HuggingFace and tokenizes using the CodeLingual tokenizer.
    """

    def __init__(
        self,
        split: str = "train",
        seq_len: int = 128,
        vocab_size: int = 48000,
        tokenizer: Optional[object] = None,
    ):
        from datasets import load_dataset

        from nexus.tokenizer.codelingual import CodeLingualTokenizer

        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # Use provided tokenizer or create default
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = CodeLingualTokenizer(vocab_size=vocab_size)

        # Download WikiText-2
        print(f"Loading WikiText-2 ({split} split)...")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

        # Tokenize all text into one long token sequence
        print("Tokenizing...")
        all_tokens: list[int] = []
        for sample in ds:
            text = sample["text"]
            if not text or text.isspace():
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(ids)

        print(f"Total tokens: {len(all_tokens):,}")

        # Chunk into fixed-length sequences
        num_chunks = len(all_tokens) // seq_len
        all_tokens = all_tokens[: num_chunks * seq_len]
        self.data = torch.tensor(all_tokens, dtype=torch.long).view(num_chunks, seq_len)
        print(f"Created {len(self.data):,} sequences of length {seq_len}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids = self.data[idx]
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100
        return {"input_ids": input_ids, "labels": labels}


class TinyStoriesDataset(Dataset):
    """
    TinyStories dataset — small, clean English stories for LM training.
    Good for verifying language learning on small models.
    """

    def __init__(
        self,
        split: str = "train",
        seq_len: int = 128,
        vocab_size: int = 48000,
        max_samples: int = 50000,
        tokenizer: Optional[object] = None,
    ):
        from datasets import load_dataset

        from nexus.tokenizer.codelingual import CodeLingualTokenizer

        self.seq_len = seq_len
        self.vocab_size = vocab_size

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = CodeLingualTokenizer(vocab_size=vocab_size)

        # Download TinyStories
        print(f"Loading TinyStories ({split} split, max {max_samples:,} samples)...")
        ds = load_dataset("roneneldan/TinyStories", split=split)

        # Tokenize into one long sequence
        print("Tokenizing...")
        all_tokens: list[int] = []
        count = 0
        for sample in ds:
            text = sample["text"]
            if not text or text.isspace():
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(ids)
            count += 1
            if count >= max_samples:
                break

        print(f"Processed {count:,} stories, total tokens: {len(all_tokens):,}")

        # Chunk into fixed-length sequences
        num_chunks = len(all_tokens) // seq_len
        all_tokens = all_tokens[: num_chunks * seq_len]
        self.data = torch.tensor(all_tokens, dtype=torch.long).view(num_chunks, seq_len)
        print(f"Created {len(self.data):,} sequences of length {seq_len}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids = self.data[idx]
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100
        return {"input_ids": input_ids, "labels": labels}


class CodeSearchNetDataset(Dataset):
    """
    CodeSearchNet dataset for code modeling.
    Uses Python subset for code-aware training.
    """

    def __init__(
        self,
        split: str = "train",
        seq_len: int = 128,
        vocab_size: int = 48000,
        max_samples: int = 50000,
        language: str = "python",
        tokenizer: Optional[object] = None,
    ):
        from datasets import load_dataset

        from nexus.tokenizer.codelingual import CodeLingualTokenizer

        self.seq_len = seq_len
        self.vocab_size = vocab_size

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = CodeLingualTokenizer(vocab_size=vocab_size)

        # Download CodeSearchNet
        print(f"Loading CodeSearchNet ({language}, {split} split)...")
        ds = load_dataset("code_search_net", language, split=split, trust_remote_code=True)

        # Tokenize code into one long sequence
        print("Tokenizing code...")
        all_tokens: list[int] = []
        count = 0
        for sample in ds:
            code = sample.get("whole_func_string", sample.get("func_code_string", ""))
            if not code or code.isspace():
                continue
            ids = self.tokenizer.encode(code, add_special_tokens=False)
            all_tokens.extend(ids)
            count += 1
            if count >= max_samples:
                break

        print(f"Processed {count:,} code samples, total tokens: {len(all_tokens):,}")

        # Chunk into fixed-length sequences
        num_chunks = len(all_tokens) // seq_len
        if num_chunks == 0:
            raise ValueError("Not enough tokens to create even one sequence")
        all_tokens = all_tokens[: num_chunks * seq_len]
        self.data = torch.tensor(all_tokens, dtype=torch.long).view(num_chunks, seq_len)
        print(f"Created {len(self.data):,} sequences of length {seq_len}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids = self.data[idx]
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100
        return {"input_ids": input_ids, "labels": labels}
