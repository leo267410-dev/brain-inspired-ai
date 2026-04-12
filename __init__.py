"""
Brain-Inspired AI Architecture
A comprehensive neural network system modeled after the human brain's structure and neuronal diversity.

This package implements:
- Diverse neuron types (excitatory, inhibitory, modulatory)
- Hierarchical brain organization (cortex, hippocampus, basal ganglia, cerebellum, thalamus)
- Brain-inspired learning mechanisms (Hebbian, dopamine-modulated, error-based)
- Comprehensive demonstration and testing framework

Based on the comprehensive neuronal taxonomy identifying over 3,000 distinct neuronal types
in the human brain and modern single-cell transcriptomic research.
"""

__version__ = "1.0.0"
__author__ = "Brain-Inspired AI Research Team"

# Core components
from .core_architecture import (
    BrainInspiredAI,
    create_brain_model,
    BaseNeuron,
    PyramidalNeuron,
    ParvalbuminInterneuron,
    SomatostatinInterneuron,
    DopaminergicNeuron,
    CorticalColumn,
    Thalamus,
    Hippocampus,
    BasalGanglia,
    Cerebellum
)

# Learning mechanisms
from .learning_mechanisms import (
    BrainInspiredTrainer,
    create_trainer,
    LearningParameters,
    HebbianLearning,
    DopamineModulatedLearning,
    CerebellarLearning,
    HippocampalLearning,
    NeuromodulatorySystem,
    AdaptiveLearningRate
)

# Demo framework
from .demo_framework import (
    MNISTDemo,
    ReinforcementLearningDemo,
    MemoryDemo,
    BenchmarkSuite,
    DemoConfig
)

__all__ = [
    # Core architecture
    "BrainInspiredAI",
    "create_brain_model",
    "BaseNeuron",
    "PyramidalNeuron",
    "ParvalbuminInterneuron",
    "SomatostatinInterneuron",
    "DopaminergicNeuron",
    "CorticalColumn",
    "Thalamus",
    "Hippocampus",
    "BasalGanglia",
    "Cerebellum",
    
    # Learning mechanisms
    "BrainInspiredTrainer",
    "create_trainer",
    "LearningParameters",
    "HebbianLearning",
    "DopamineModulatedLearning",
    "CerebellarLearning",
    "HippocampalLearning",
    "NeuromodulatorySystem",
    "AdaptiveLearningRate",
    
    # Demo framework
    "MNISTDemo",
    "ReinforcementLearningDemo",
    "MemoryDemo",
    "BenchmarkSuite",
    "DemoConfig",
]
