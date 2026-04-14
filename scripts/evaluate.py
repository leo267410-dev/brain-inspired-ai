#!/usr/bin/env python3
"""Evaluation script for NEXUS-Ω."""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexus.config import nexus_omega_small
from nexus.data.synthetic import SyntheticLanguageDataset
from nexus.model.nexus_model import NexusOmegaModel
from nexus.utils.metrics import MetricTracker, compute_accuracy, compute_perplexity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NEXUS-Ω model")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = nexus_omega_small()
    model = NexusOmegaModel(config)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {args.checkpoint}")

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = SyntheticLanguageDataset(
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        vocab_size=config.vocab_size,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    tracker = MetricTracker()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids)
            logits = outputs["logits"]

            ppl = compute_perplexity(logits[:, :-1], labels[:, :logits.size(1) - 1])
            acc = compute_accuracy(logits[:, :-1], labels[:, :logits.size(1) - 1])

            tracker.update("perplexity", ppl)
            tracker.update("accuracy", acc)

    summary = tracker.summary()
    print("Evaluation Results:")
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
