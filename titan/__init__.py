"""
TITAN — Tiled Interleaved Transformer with Adaptive Neurons
============================================================
A sub-150M parameter language model designed for massive GPU parallelism.

Key innovations:
  1. Conformal GQA — distance-based attention scoring (+16% cluster alignment, zero extra params)
  2. Oscillatory Expert Banks — SRN activation in MoE experts for richer per-parameter compute
  3. Hybrid Recurrent-Attention — O(1) recurrent mixer + sparse global attention anchors
  4. Adaptive Difficulty Routing — scratchpad refinement only for hard tokens
  5. Multi-Token Prediction — 4 auxiliary heads for faster training + speculative decoding
"""

from titan.config import TitanConfig
from titan.model import Titan

__all__ = ["TitanConfig", "Titan"]
__version__ = "0.1.0"
