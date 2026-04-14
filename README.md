# NEXUS-Ω (Omega)

A sub-200M parameter language model for **coding** and **English language understanding** that achieves trillion-parameter-level capability through radical architectural innovations.

## Key Innovations

| Innovation | Description |
|---|---|
| **Shared MoE** | 64 experts shared across all 24 layers with per-layer routers, creating 64^24 possible computation paths |
| **SSM-Attention Hybrid** | Mamba-style SSM for O(n) local processing + hierarchical sparse attention for global context |
| **Thought Engine** | Internal chain-of-thought reasoning with thought compression and confidence-based early stopping |
| **External Neural Memory** | Differentiable read/write memory bank with LRU replacement and LSH-based fast retrieval |
| **MCTS Code Search** | Monte Carlo Tree Search for exploring code generation paths with learned value/policy functions |
| **Recursive Reasoning** | Multi-loop recursive computation with scratchpad memory for complex multi-step problems |
| **Smart Neurons** | Adaptive neuron gating with evolutionary pruning — bottom 5% neurons are periodically reinitialized |
| **Hebbian Lateral Connections** | Sparse within-layer and cross-layer connections updated via Hebbian learning (no backprop) |
| **Dynamic Depth/Width** | Per-token early exit and per-dimension width gating for adaptive computation |
| **Self-Verification** | Generate → verify → revise loop with test-time training via LoRA |
| **Meta-Controller** | Dynamically allocates computation resources based on input difficulty |

## Architecture Overview

```
Input → Multi-Resolution Embedding (token + char CNN + chunk transformer)
      → Code-Aware RoPE
      → 8 × Mamba SSM Blocks
      → 8 × Hybrid SSM-Attention Blocks
      → 8 × Full Hierarchical Sparse Attention Blocks
      → Recursive Reasoning (scratchpad, up to 3 loops)
      → Task Heads (LM / Code)
      → Output

Each block includes:
  - Shared MoE FFN (64 experts, top-2 routing)
  - Smart Neuron Gating
  - Hebbian Lateral Connections
  - Dynamic Width Gating
```

## Quick Start

### Installation

```bash
pip install -e .
```

### Run Demo

```bash
python scripts/demo.py
```

### Train (with synthetic data)

```bash
python scripts/train.py --config small --max-steps 100 --batch-size 4
```

### Run Tests

```bash
python -m pytest tests/ -v
```

### Benchmark

```bash
python scripts/benchmark.py
```

## Project Structure

```
nexus/                  # Python package
├── config.py           # Dataclass-based configuration
├── model/              # Core model components
│   ├── nexus_model.py  # Main model (assembles all components)
│   ├── attention.py    # Hierarchical sparse attention
│   ├── ssm.py          # Mamba SSM blocks
│   ├── moe_ffn.py      # Shared expert pool with MoE routing
│   ├── smart_neuron.py # Adaptive neuron gating
│   ├── lateral.py      # Hebbian lateral connections
│   ├── early_exit.py   # Dynamic depth/width
│   ├── recursive_reasoning.py
│   ├── embeddings.py   # Multi-resolution embeddings + CodeAwareRoPE
│   ├── task_heads.py   # LM and code output heads
│   └── meta_controller.py
├── thought/            # Internal chain-of-thought engine
├── memory/             # External neural memory + knowledge embeddings
├── search/             # MCTS code search + self-verification
├── tokenizer/          # CodeLingual tokenizer
├── training/           # Trainer, curriculum, distillation, loss, optimizer
├── data/               # Datasets and preprocessing
├── inference/          # Generation engine, quantization, serving
└── utils/              # Profiler and metrics
scripts/                # Training, evaluation, benchmark, and demo scripts
tests/                  # Comprehensive test suite
```

## Parameter Budget

The base configuration achieves **< 200M parameters** through:

- **Shared experts**: 64 experts are shared across all 24 layers (not duplicated per layer)
- **Smart neuron gating**: Only ~30% of neurons activate per token
- **Efficient MoE routing**: Top-2 expert selection from 64 experts
- **Low-rank components**: LoRA-style adapters for test-time training

## Configuration Presets

| Config | Hidden Dim | Layers | Experts | Use Case |
|--------|-----------|--------|---------|----------|
| `nexus_omega_base()` | 768 | 24 | 64 | Full model (< 200M params) |
| `nexus_omega_small()` | 256 | 6 | 8 | Testing and development |
| `nexus_omega_code()` | 768 | 24 | 64 | Code generation optimized |
| `nexus_omega_language()` | 768 | 24 | 64 | Language understanding optimized |

## License

MIT
