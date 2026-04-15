"""TITAN layer components."""

from titan.layers.activations import OscillatoryActivation
from titan.layers.attention import ConformalGQA
from titan.layers.embeddings import RotaryEmbedding, apply_rope
from titan.layers.moe import OscillatoryExpertBank, SmartMoEFFN
from titan.layers.norms import RMSNorm
from titan.layers.recurrent import RecurrentMixer
from titan.layers.scratchpad import DifficultyGate, ScratchpadRefiner

__all__ = [
    "OscillatoryActivation",
    "ConformalGQA",
    "RotaryEmbedding",
    "apply_rope",
    "OscillatoryExpertBank",
    "SmartMoEFFN",
    "RMSNorm",
    "RecurrentMixer",
    "DifficultyGate",
    "ScratchpadRefiner",
]
