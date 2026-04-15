"""TITAN parameter budget estimator.

Calculates exact parameter counts for each component to verify
the model stays under the 150M parameter budget.
"""

from __future__ import annotations

from titan.config import TitanConfig


def estimate_params(cfg: TitanConfig) -> dict:
    """Estimate total parameter count from config."""
    d = cfg.d_model
    vocab = cfg.vocab_size
    kv_dim = cfg.kv_dim
    h = cfg.bank_hidden
    n_banks = cfg.routing_banks

    # Token embedding (shared with LM head via tying)
    embedding = vocab * d

    # Per-layer components (all layers)
    # Recurrent mixer: proj(d→4d) + dw_conv(d×k) + out(d→d)
    recurrent_per_layer = (d * 4 * d) + (d * cfg.recurrent_kernel_size) + (d * d)

    # Difficulty gate: Linear(d→1) + bias
    diff_gate_per_layer = d + 1

    # Scratchpad: slots(n×d) + score(d→n) + mix(2d→bottleneck) + update(bottleneck→2d)
    scratch_per_layer = (
        cfg.scratch_slots * d
        + d * cfg.scratch_slots
        + (2 * d) * cfg.scratch_bottleneck
        + cfg.scratch_bottleneck * (2 * d)
    )

    # Router: Linear(d→n_banks) + bias
    router_per_layer = d * n_banks + n_banks

    # RMS norms: up to 3 per layer (recurrent, attention, ffn)
    norms_per_layer = 3 * d  # attention layers have 3 norms

    # MoE FFN output norm
    moe_norm_per_layer = d

    # Total per non-attention layer
    base_per_layer = (
        recurrent_per_layer
        + diff_gate_per_layer
        + scratch_per_layer
        + router_per_layer
        + norms_per_layer
        + moe_norm_per_layer
    )

    # Attention (anchor layers only): Q(d→d) + K(d→kv) + V(d→kv) + O(d→d)
    attention_per_anchor = d * d + d * kv_dim + d * kv_dim + d * d

    n_anchors = cfg.n_anchor_layers

    # Shared expert banks (oscillatory or SwiGLU, both use 2 up + 1 down)
    if cfg.use_oscillatory:
        shared_banks = n_banks * (d * h + d * h + h * d) + n_banks * h  # +phase
    else:
        shared_banks = n_banks * (d * h + d * h + h * d)

    # Multi-token prediction heads: n × (norm(d) + linear(d→d) + GELU implicit)
    mtp = cfg.mtp_heads * (d + d * d)

    # Final norm
    final_norm = d

    # LM head (tied with embedding → 0 additional)
    lm_head = 0 if cfg.tie_embeddings else vocab * d

    total = (
        embedding
        + cfg.n_layers * base_per_layer
        + n_anchors * attention_per_anchor
        + shared_banks
        + mtp
        + final_norm
        + lm_head
    )

    return {
        "config_summary": {
            "vocab_size": vocab,
            "d_model": d,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "n_kv_heads": cfg.n_kv_heads,
            "n_anchor_layers": n_anchors,
            "routing_banks": n_banks,
            "bank_hidden": h,
            "use_oscillatory": cfg.use_oscillatory,
        },
        "estimated_total_params": total,
        "estimated_total_millions": round(total / 1_000_000, 2),
        "under_150m": total < 150_000_000,
        "breakdown": {
            "embedding": embedding,
            "layer_stack": cfg.n_layers * base_per_layer,
            "attention_anchors": n_anchors * attention_per_anchor,
            "shared_expert_banks": shared_banks,
            "multi_token_heads": mtp,
            "final_norm": final_norm,
            "lm_head": lm_head,
        },
    }


if __name__ == "__main__":
    cfg = TitanConfig()
    report = estimate_params(cfg)

    print("=" * 60)
    print("TITAN Parameter Budget Report")
    print("=" * 60)
    print(f"\nTotal parameters: {report['estimated_total_millions']}M")
    print(f"Under 150M budget: {report['under_150m']}")
    print("\nBreakdown:")
    for key, value in report["breakdown"].items():
        millions = round(value / 1e6, 2)
        pct = round(100 * value / report["estimated_total_params"], 1)
        print(f"  {key:.<30} {millions:>8.2}M  ({pct}%)")

    print("\nConfig:")
    for key, value in report["config_summary"].items():
        print(f"  {key}: {value}")
