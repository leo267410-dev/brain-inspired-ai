"""Curriculum learning strategies for progressive training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from torch.utils.data import Dataset, Subset


class DifficultyLevel(Enum):
    """Difficulty levels for curriculum learning."""

    EASY = 1
    MEDIUM = 2
    HARD = 3
    EXPERT = 4


@dataclass
class CurriculumStage:
    """A single stage in the curriculum."""

    name: str
    difficulty: DifficultyLevel
    max_seq_len: int
    num_steps: int
    tasks: List[str]  # e.g., ["lm", "code", "reasoning"]
    description: str = ""


class CurriculumScheduler:
    """
    Manages progressive curriculum learning.
    Starts with simple tasks and gradually increases difficulty.
    """

    def __init__(self, stages: Optional[List[CurriculumStage]] = None):
        if stages is None:
            stages = self.default_curriculum()
        self.stages = stages
        self.current_stage_idx = 0
        self.steps_in_stage = 0

    @staticmethod
    def default_curriculum() -> List[CurriculumStage]:
        """Create the default 4-stage curriculum."""
        return [
            CurriculumStage(
                name="foundation",
                difficulty=DifficultyLevel.EASY,
                max_seq_len=512,
                num_steps=50_000,
                tasks=["lm"],
                description="Basic language modeling on short sequences",
            ),
            CurriculumStage(
                name="expansion",
                difficulty=DifficultyLevel.MEDIUM,
                max_seq_len=2048,
                num_steps=100_000,
                tasks=["lm", "code"],
                description="Longer sequences with code understanding",
            ),
            CurriculumStage(
                name="specialization",
                difficulty=DifficultyLevel.HARD,
                max_seq_len=4096,
                num_steps=200_000,
                tasks=["lm", "code", "reasoning"],
                description="Complex tasks with reasoning chains",
            ),
            CurriculumStage(
                name="mastery",
                difficulty=DifficultyLevel.EXPERT,
                max_seq_len=8192,
                num_steps=150_000,
                tasks=["lm", "code", "reasoning"],
                description="Full difficulty with all features enabled",
            ),
        ]

    @property
    def current_stage(self) -> CurriculumStage:
        """Get the current curriculum stage."""
        return self.stages[self.current_stage_idx]

    def step(self) -> bool:
        """
        Advance one step. Returns True if stage changed.
        """
        self.steps_in_stage += 1
        if self.steps_in_stage >= self.current_stage.num_steps:
            if self.current_stage_idx < len(self.stages) - 1:
                self.current_stage_idx += 1
                self.steps_in_stage = 0
                return True
        return False

    def get_max_seq_len(self) -> int:
        """Get the current maximum sequence length."""
        return self.current_stage.max_seq_len

    def get_tasks(self) -> List[str]:
        """Get the current active tasks."""
        return self.current_stage.tasks

    def filter_dataset(self, dataset: Dataset, max_len: Optional[int] = None) -> Subset:
        """
        Filter dataset to only include samples appropriate for current stage.

        Args:
            dataset: full dataset.
            max_len: override max sequence length.

        Returns:
            Filtered subset.
        """
        if max_len is None:
            max_len = self.get_max_seq_len()

        valid_indices = []
        for i in range(len(dataset)):
            sample = dataset[i]
            if isinstance(sample, dict) and "input_ids" in sample:
                if len(sample["input_ids"]) <= max_len:
                    valid_indices.append(i)
            else:
                valid_indices.append(i)

        return Subset(dataset, valid_indices)
