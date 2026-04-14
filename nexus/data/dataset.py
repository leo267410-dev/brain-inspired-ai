"""Dataset classes for NEXUS-Ω training."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset


class NexusDataset(Dataset):
    """
    General-purpose dataset for language modeling and code generation.
    Supports both pre-tokenized data and on-the-fly tokenization.
    """

    def __init__(
        self,
        data: Union[List[Dict], str],
        max_seq_len: int = 8192,
        tokenizer: Optional[object] = None,
    ):
        """
        Args:
            data: list of dicts with 'input_ids' or path to data file.
            max_seq_len: maximum sequence length.
            tokenizer: optional tokenizer for on-the-fly tokenization.
        """
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

        if isinstance(data, str):
            self.samples = self._load_from_file(data)
        else:
            self.samples = data

    def _load_from_file(self, path: str) -> List[Dict]:
        """Load data from a JSONL or text file."""
        import json

        samples = []
        file_path = Path(path)

        if file_path.suffix == ".jsonl":
            with open(file_path) as f:
                for line in f:
                    samples.append(json.loads(line.strip()))
        elif file_path.suffix == ".txt":
            with open(file_path) as f:
                text = f.read()
            # Split into chunks
            chunk_size = self.max_seq_len
            for i in range(0, len(text), chunk_size):
                chunk = text[i: i + chunk_size]
                if self.tokenizer is not None:
                    ids = self.tokenizer.encode(chunk, max_length=self.max_seq_len)
                    samples.append({"input_ids": ids})
                else:
                    samples.append({"text": chunk})
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        if "input_ids" in sample:
            input_ids = sample["input_ids"]
            if isinstance(input_ids, list):
                input_ids = torch.tensor(input_ids, dtype=torch.long)
        elif "text" in sample and self.tokenizer is not None:
            input_ids = torch.tensor(
                self.tokenizer.encode(sample["text"], max_length=self.max_seq_len),
                dtype=torch.long,
            )
        else:
            raise ValueError("Sample must have 'input_ids' or 'text' with a tokenizer")

        # Truncate
        input_ids = input_ids[: self.max_seq_len]

        # Labels are shifted input_ids
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100  # Ignore last token

        return {"input_ids": input_ids, "labels": labels}


class CodeDataset(NexusDataset):
    """Dataset specialized for code, with structure-aware tokenization."""

    def __init__(
        self,
        data: Union[List[Dict], str],
        max_seq_len: int = 8192,
        tokenizer: Optional[object] = None,
        language: str = "python",
    ):
        super().__init__(data, max_seq_len, tokenizer)
        self.language = language

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        result = super().__getitem__(idx)

        # Add segment IDs (0 = code)
        result["segment_ids"] = torch.zeros_like(result["input_ids"])

        return result


class InterleavedDataset(Dataset):
    """
    Interleaves multiple datasets (code + text) with configurable ratios.
    """

    def __init__(
        self,
        datasets: List[Dataset],
        ratios: Optional[List[float]] = None,
        total_size: int = 100_000,
    ):
        self.datasets = datasets
        if ratios is None:
            ratios = [1.0 / len(datasets)] * len(datasets)
        self.ratios = ratios
        self.total_size = total_size

        # Precompute index mapping
        self._indices: List[tuple[int, int]] = []
        for i in range(total_size):
            # Sample from datasets according to ratios
            r = (i * 0.618033988749895) % 1.0  # Low-discrepancy sequence
            cumulative = 0.0
            for ds_idx, ratio in enumerate(self.ratios):
                cumulative += ratio
                if r < cumulative:
                    sample_idx = i % len(self.datasets[ds_idx])
                    self._indices.append((ds_idx, sample_idx))
                    break

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ds_idx, sample_idx = self._indices[idx]
        return self.datasets[ds_idx][sample_idx]
