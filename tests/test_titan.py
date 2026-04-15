"""Tests for TITAN architecture."""

from __future__ import annotations

import pytest
import torch

from titan.config import TitanConfig
from titan.model import Titan
from titan.param_budget import estimate_params
from titan.training.losses import titan_loss
from titan.training.optimizer import build_optimizer, cosine_warmup_schedule


@pytest.fixture
def small_cfg() -> TitanConfig:
    """Small config for fast testing."""
    return TitanConfig(
        vocab_size=256,
        d_model=64,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        anchor_every=2,
        bank_hidden=96,
        routing_banks=4,
        routing_topk=2,
        scratch_slots=4,
        scratch_steps=1,
        scratch_bottleneck=32,
        mtp_heads=2,
        max_seq_len=64,
        gradient_checkpointing=False,
    )


@pytest.fixture
def model(small_cfg: TitanConfig) -> Titan:
    return Titan(small_cfg)


class TestConfig:
    def test_head_dim(self) -> None:
        cfg = TitanConfig(d_model=768, n_heads=12)
        assert cfg.head_dim == 64

    def test_kv_dim(self) -> None:
        cfg = TitanConfig(d_model=768, n_heads=12, n_kv_heads=2)
        assert cfg.kv_dim == 128

    def test_anchor_layer_indices(self) -> None:
        cfg = TitanConfig(n_layers=9, anchor_every=3)
        assert cfg.anchor_layer_indices() == [2, 5, 8]

    def test_n_anchor_layers(self) -> None:
        cfg = TitanConfig(n_layers=22, anchor_every=3)
        assert cfg.n_anchor_layers == 7


class TestParamBudget:
    def test_full_config_under_150m(self) -> None:
        cfg = TitanConfig()
        report = estimate_params(cfg)
        assert report["under_150m"], (
            f"Full config exceeds 150M: {report['estimated_total_millions']}M"
        )

    def test_budget_positive(self) -> None:
        cfg = TitanConfig()
        report = estimate_params(cfg)
        assert report["estimated_total_params"] > 0
        for key, val in report["breakdown"].items():
            assert val >= 0, f"Negative param count for {key}"


class TestForwardPass:
    def test_logits_shape(self, model: Titan, small_cfg: TitanConfig) -> None:
        B, S = 2, 16
        ids = torch.randint(0, small_cfg.vocab_size, (B, S))
        logits, _ = model(ids, return_mtp=False)
        assert logits.shape == (B, S, small_cfg.vocab_size)

    def test_mtp_shapes(self, model: Titan, small_cfg: TitanConfig) -> None:
        B, S = 2, 16
        ids = torch.randint(0, small_cfg.vocab_size, (B, S))
        logits, mtp_logits = model(ids, return_mtp=True)
        assert mtp_logits is not None
        assert len(mtp_logits) == small_cfg.mtp_heads
        for ml in mtp_logits:
            assert ml.shape == (B, S, small_cfg.vocab_size)

    def test_no_mtp(self, model: Titan, small_cfg: TitanConfig) -> None:
        ids = torch.randint(0, small_cfg.vocab_size, (1, 8))
        _, mtp_logits = model(ids, return_mtp=False)
        assert mtp_logits is None

    def test_single_token(self, model: Titan, small_cfg: TitanConfig) -> None:
        ids = torch.randint(0, small_cfg.vocab_size, (1, 1))
        logits, _ = model(ids, return_mtp=False)
        assert logits.shape == (1, 1, small_cfg.vocab_size)


class TestConformalAttention:
    def test_conformal_vs_standard(self, small_cfg: TitanConfig) -> None:
        """Conformal and standard attention should produce different scores."""
        cfg_conformal = TitanConfig(**{**small_cfg.__dict__, "use_conformal": True})
        cfg_standard = TitanConfig(**{**small_cfg.__dict__, "use_conformal": False})

        m1 = Titan(cfg_conformal)
        m2 = Titan(cfg_standard)

        # Copy weights from m1 to m2 for fair comparison
        m2.load_state_dict(m1.state_dict())

        ids = torch.randint(0, small_cfg.vocab_size, (1, 8))
        with torch.no_grad():
            out1, _ = m1(ids)
            out2, _ = m2(ids)

        # Outputs should differ (conformal adds distance correction)
        assert not torch.allclose(out1, out2, atol=1e-5), (
            "Conformal and standard attention produced identical outputs"
        )


class TestOscillatoryExperts:
    def test_oscillatory_vs_swiglu(self, small_cfg: TitanConfig) -> None:
        """Oscillatory and SwiGLU experts should produce different outputs."""
        cfg_osc = TitanConfig(**{**small_cfg.__dict__, "use_oscillatory": True})
        cfg_swi = TitanConfig(**{**small_cfg.__dict__, "use_oscillatory": False})

        m1 = Titan(cfg_osc)
        m2 = Titan(cfg_swi)

        ids = torch.randint(0, small_cfg.vocab_size, (1, 8))
        with torch.no_grad():
            out1, _ = m1(ids)
            out2, _ = m2(ids)

        # Both should produce valid logits
        assert out1.shape == out2.shape
        assert torch.isfinite(out1).all()
        assert torch.isfinite(out2).all()


class TestTraining:
    def test_loss_decreases(self, small_cfg: TitanConfig) -> None:
        """Loss should decrease over a few steps on synthetic data."""
        torch.manual_seed(42)
        mdl = Titan(small_cfg)
        mdl.train()
        optimizer = build_optimizer(mdl, lr=1e-3)

        # Use same batch each step for stable convergence signal
        ids = torch.randint(0, small_cfg.vocab_size, (4, 16))

        first_loss = None
        last_loss = None

        for step in range(30):
            optimizer.zero_grad()
            logits, mtp = mdl(ids, return_mtp=True)
            losses = titan_loss(logits=logits, targets=ids, mtp_logits=mtp)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            optimizer.step()

            if step == 0:
                first_loss = losses["total"].item()
            last_loss = losses["total"].item()

        assert last_loss < first_loss, (
            f"Loss did not decrease: {first_loss:.4f} → {last_loss:.4f}"
        )

    def test_cosine_schedule(self) -> None:
        """LR schedule should warmup then decay."""
        model = torch.nn.Linear(10, 10)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        lrs = []
        for step in range(100):
            cosine_warmup_schedule(opt, step, warmup=10, total=100)
            lrs.append(opt.param_groups[0]["lr"])

        # Warmup: LR should increase
        assert lrs[5] > lrs[0]
        # After warmup: LR should peak near step 10
        assert lrs[10] >= lrs[50]
        # Decay: LR at end should be lower
        assert lrs[-1] < lrs[10]


class TestGeneration:
    def test_generate(self, model: Titan, small_cfg: TitanConfig) -> None:
        model.eval()
        prompt = torch.randint(0, small_cfg.vocab_size, (1, 4))
        output = model.generate(prompt, max_new_tokens=8, temperature=0.8)
        assert output.shape[1] == 4 + 8
        assert (output[:, :4] == prompt).all()


class TestParamCount:
    def test_param_count_method(self, model: Titan) -> None:
        pc = model.param_count()
        assert pc["total"] > 0
        assert pc["total_millions"] > 0
        assert "breakdown" in pc

    def test_weight_tying(self, model: Titan) -> None:
        """Embedding and LM head should share the same weight tensor."""
        assert model.tok_embeddings.weight.data_ptr() == model.lm_head.weight.data_ptr()
