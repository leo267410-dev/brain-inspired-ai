"""Hierarchical sparse attention with local windows and global landmarks."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from nexus.config import NexusOmegaConfig

from nexus.model.embeddings import CodeAwareRoPE


class LocalWindowAttention(nn.Module):
    """Sliding-window attention with chunked computation."""

    def __init__(self, window_size: int = 256):
        super().__init__()
        self.window_size = window_size

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            q, k, v: (batch, seq_len, num_heads, head_dim)
            attention_mask: optional mask

        Returns:
            (batch, seq_len, num_heads, head_dim)
        """
        B, L, H, D = q.shape
        w = self.window_size

        # Pad sequence to multiple of window_size
        pad_len = (w - L % w) % w
        if pad_len > 0:
            q = F.pad(q, (0, 0, 0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, 0, 0, pad_len))

        L_padded = q.shape[1]
        num_chunks = L_padded // w

        # Reshape into windows
        q_win = q.view(B, num_chunks, w, H, D)
        k_win = k.view(B, num_chunks, w, H, D)
        v_win = v.view(B, num_chunks, w, H, D)

        # Compute attention within each window
        # (B, num_chunks, H, w, D) @ (B, num_chunks, H, D, w) -> (B, num_chunks, H, w, w)
        q_win = q_win.permute(0, 1, 3, 2, 4)  # (B, nc, H, w, D)
        k_win = k_win.permute(0, 1, 3, 2, 4)
        v_win = v_win.permute(0, 1, 3, 2, 4)

        scale = math.sqrt(D)
        attn_scores = torch.matmul(q_win, k_win.transpose(-2, -1)) / scale

        # Causal mask within window
        causal = torch.triu(
            torch.ones(w, w, device=q.device, dtype=torch.bool), diagonal=1,
        )
        attn_scores = attn_scores.masked_fill(causal[None, None, None], float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v_win)  # (B, nc, H, w, D)

        # Reshape back
        out = out.permute(0, 1, 3, 2, 4).reshape(B, L_padded, H, D)
        return out[:, :L, :, :]


class GlobalLandmarkAttention(nn.Module):
    """
    Sparse global attention via landmark tokens.
    Every landmark_interval tokens is a landmark.
    All tokens attend to all landmarks.
    """

    def __init__(self, landmark_interval: int = 64):
        super().__init__()
        self.landmark_interval = landmark_interval

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        code_anchors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            q, k, v: (batch, seq_len, num_heads, head_dim)
            attention_mask: optional mask
            code_anchors: optional (batch, num_anchors) position indices

        Returns:
            (batch, seq_len, num_heads, head_dim)
        """
        B, L, H, D = q.shape

        # Extract landmark positions
        landmark_positions = torch.arange(0, L, self.landmark_interval, device=q.device)

        # Add code anchors as extra landmarks if provided
        if code_anchors is not None:
            # code_anchors: (B, num_anchors)
            # For simplicity, use the first batch's anchors as shared landmarks
            extra = code_anchors[0][code_anchors[0] < L].unique()
            landmark_positions = torch.cat([landmark_positions, extra]).unique().sort().values

        # Gather landmark K, V
        landmark_k = k[:, landmark_positions]  # (B, num_landmarks, H, D)
        landmark_v = v[:, landmark_positions]  # (B, num_landmarks, H, D)

        # All tokens attend to landmarks
        q_t = q.permute(0, 2, 1, 3)  # (B, H, L, D)
        lk_t = landmark_k.permute(0, 2, 1, 3)  # (B, H, num_landmarks, D)
        lv_t = landmark_v.permute(0, 2, 1, 3)  # (B, H, num_landmarks, D)

        scale = math.sqrt(D)
        attn_scores = torch.matmul(q_t, lk_t.transpose(-2, -1)) / scale  # (B, H, L, NL)
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, lv_t)  # (B, H, L, D)

        return out.permute(0, 2, 1, 3)  # (B, L, H, D)


class HierarchicalSparseAttention(nn.Module):
    """
    Combines local window attention with global landmark attention
    and code-aware rotary position embeddings.
    """

    def __init__(self, config: NexusOmegaConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_dim = config.hidden_dim

        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

        self.local_window = LocalWindowAttention(config.local_window_size)
        self.global_landmark = GlobalLandmarkAttention(config.global_landmark_interval)

        # Learned gate to blend local and global
        self.blend_gate = nn.Parameter(torch.tensor(0.5))

        self.rope = CodeAwareRoPE(config.head_dim, config.max_seq_len)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        code_anchors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            attention_mask: optional mask
            kv_cache: optional (cached_k, cached_v) tuple
            code_anchors: optional tensor of important positions

        Returns:
            (batch, seq_len, hidden_dim)
        """
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)

        # Apply RoPE
        q, k = self.rope(q, k)

        # Handle KV cache for autoregressive generation
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=1)
            v = torch.cat([cached_v, v], dim=1)

        # Local attention
        local_out = self.local_window(q, k, v, attention_mask)

        # Global landmark attention
        global_out = self.global_landmark(q, k, v, attention_mask, code_anchors)

        # Blend local and global
        gate = torch.sigmoid(self.blend_gate)
        out = gate * local_out + (1 - gate) * global_out

        out = out.reshape(B, L, D)
        return self.o_proj(self.dropout(out))
