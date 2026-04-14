#!/usr/bin/env python3
"""Training script for NEXUS-Ω."""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexus.config import (
    nexus_omega_base,
    nexus_omega_code,
    nexus_omega_language,
    nexus_omega_small,
)
from nexus.model.nexus_model import NexusOmegaModel
from nexus.training.optimizer import NexusOptimizer, create_cosine_schedule_with_warmup
from nexus.training.trainer import NexusTrainer, TrainingArgs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NEXUS-Ω model")
    parser.add_argument("--config", type=str, default="small", choices=["base", "small", "code", "language"])
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)

    # Dataset selection
    parser.add_argument(
        "--dataset", type=str, default="wikitext",
        choices=["synthetic", "wikitext", "tinystories", "codesearchnet"],
        help="Dataset to use for training",
    )
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples for synthetic dataset")
    parser.add_argument("--max-dataset-samples", type=int, default=50000, help="Max samples for real datasets")
    return parser.parse_args()


def create_dataset(args: argparse.Namespace, vocab_size: int) -> torch.utils.data.Dataset:
    """Create dataset based on args."""
    if args.dataset == "synthetic":
        from nexus.data.synthetic import SyntheticLanguageDataset
        return SyntheticLanguageDataset(
            num_samples=args.num_samples,
            seq_len=args.seq_len,
            vocab_size=vocab_size,
            seed=args.seed,
        )
    elif args.dataset == "wikitext":
        from nexus.data.real_datasets import WikiTextDataset
        return WikiTextDataset(
            split="train",
            seq_len=args.seq_len,
            vocab_size=vocab_size,
        )
    elif args.dataset == "tinystories":
        from nexus.data.real_datasets import TinyStoriesDataset
        return TinyStoriesDataset(
            split="train",
            seq_len=args.seq_len,
            vocab_size=vocab_size,
            max_samples=args.max_dataset_samples,
        )
    elif args.dataset == "codesearchnet":
        from nexus.data.real_datasets import CodeSearchNetDataset
        return CodeSearchNetDataset(
            split="train",
            seq_len=args.seq_len,
            vocab_size=vocab_size,
            max_samples=args.max_dataset_samples,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


def main() -> None:
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Select config
    config_map = {
        "base": nexus_omega_base,
        "small": nexus_omega_small,
        "code": nexus_omega_code,
        "language": nexus_omega_language,
    }
    config = config_map[args.config]()

    print(f"Using config: {args.config}")
    print(f"Hidden dim: {config.hidden_dim}, Layers: {config.num_layers}")
    print(f"Dataset: {args.dataset}")

    # Create model
    model = NexusOmegaModel(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Create dataset
    dataset = create_dataset(args, config.vocab_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Create training args
    training_args = TrainingArgs(
        learning_rate=args.lr,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        seed=args.seed,
        log_interval=args.log_interval,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    # Create optimizer and scheduler
    optimizer = NexusOptimizer(model.parameters(), lr=args.lr)
    scheduler = create_cosine_schedule_with_warmup(
        optimizer, args.warmup_steps, args.max_steps,
    )

    # Train
    trainer = NexusTrainer(
        model=model,
        args=training_args,
        train_dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    results = trainer.train()
    print(f"Training complete. Final loss: {results['final_loss']:.4f}")


if __name__ == "__main__":
    main()
