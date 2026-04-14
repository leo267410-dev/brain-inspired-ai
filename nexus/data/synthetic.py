"""Synthetic data generation for testing and pre-training warmup."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


class SyntheticLanguageDataset(Dataset):
    """
    Generates synthetic token sequences for testing.
    Useful for verifying the training pipeline without real data.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        seq_len: int = 512,
        vocab_size: int = 48000,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # Generate deterministic random data
        generator = torch.Generator().manual_seed(seed)
        self.data = torch.randint(
            0, vocab_size, (num_samples, seq_len), generator=generator,
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids = self.data[idx]
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100
        return {"input_ids": input_ids, "labels": labels}


class SyntheticCodeDataset(Dataset):
    """
    Generates synthetic code-like token sequences.
    Includes structure tokens (indent/dedent patterns).
    """

    def __init__(
        self,
        num_samples: int = 1000,
        seq_len: int = 512,
        vocab_size: int = 48000,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        generator = torch.Generator().manual_seed(seed)
        self.data = torch.randint(
            10, vocab_size, (num_samples, seq_len), generator=generator,
        )

        # Add some structure: periodic "newline" tokens
        for i in range(num_samples):
            line_lengths = torch.randint(10, 50, (seq_len // 20,), generator=generator)
            pos = 0
            for ll in line_lengths:
                pos += ll.item()
                if pos < seq_len:
                    self.data[i, pos] = 9  # newline-like token

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids = self.data[idx]
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100
        segment_ids = torch.zeros_like(input_ids)  # all code
        return {"input_ids": input_ids, "labels": labels, "segment_ids": segment_ids}
