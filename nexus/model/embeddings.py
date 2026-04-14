"""Multi-resolution embeddings and code-aware rotary position embeddings."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharCNNEncoder(nn.Module):
    """Character-level CNN encoder for rare/OOV token handling."""

    def __init__(self, num_chars: int = 256, char_dim: int = 64, out_dim: int = 768):
        super().__init__()
        self.char_embedding = nn.Embedding(num_chars, char_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(char_dim, char_dim, kernel_size=k, padding=k // 2)
            for k in [3, 5, 7]
        ])
        self.proj = nn.Linear(char_dim * 3, out_dim)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            char_ids: (batch, seq_len, max_word_len) character indices.

        Returns:
            (batch, seq_len, out_dim) character-level embeddings.
        """
        B, L, W = char_ids.shape
        x = self.char_embedding(char_ids)  # (B, L, W, char_dim)
        x = x.view(B * L, W, -1).transpose(1, 2)  # (B*L, char_dim, W)

        conv_outs = []
        for conv in self.convs:
            c = F.relu(conv(x))  # (B*L, char_dim, W)
            c = c.max(dim=-1).values  # (B*L, char_dim)
            conv_outs.append(c)

        x = torch.cat(conv_outs, dim=-1)  # (B*L, char_dim*3)
        x = self.proj(x)  # (B*L, out_dim)
        return x.view(B, L, -1)


class ChunkTransformer(nn.Module):
    """Small transformer that produces chunk-level embeddings."""

    def __init__(self, hidden_dim: int = 768, num_layers: int = 2, num_heads: int = 4, chunk_size: int = 64):
        super().__init__()
        self.chunk_size = chunk_size
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 2,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim) token embeddings.

        Returns:
            (batch, seq_len, hidden_dim) chunk-level embeddings broadcast back.
        """
        B, L, D = x.shape
        # Pad to multiple of chunk_size
        pad_len = (self.chunk_size - L % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            x_padded = F.pad(x, (0, 0, 0, pad_len))
        else:
            x_padded = x

        num_chunks = x_padded.shape[1] // self.chunk_size
        # Reshape into chunks
        chunks = x_padded.view(B * num_chunks, self.chunk_size, D)
        chunk_out = self.encoder(chunks)  # (B*num_chunks, chunk_size, D)

        # Pool each chunk to a single vector
        chunk_repr = chunk_out.mean(dim=1)  # (B*num_chunks, D)
        chunk_repr = self.pool_proj(chunk_repr)  # (B*num_chunks, D)

        # Broadcast back to token level
        chunk_repr = chunk_repr.view(B, num_chunks, 1, D).expand(B, num_chunks, self.chunk_size, D)
        chunk_repr = chunk_repr.reshape(B, num_chunks * self.chunk_size, D)

        return chunk_repr[:, :L, :]  # Trim padding


class MultiResolutionEmbedding(nn.Module):
    """
    Multi-resolution token embeddings combining:
    - Token-level embeddings
    - Segment embeddings (code, english, reasoning)
    - Character-level CNN encoder
    - Chunk-level transformer
    - Cross-resolution fusion with learned gating
    """

    def __init__(self, vocab_size: int = 48000, hidden_dim: int = 768, max_seq_len: int = 8192):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        # Segment embedding: 0=code, 1=english, 2=reasoning
        self.segment_embedding = nn.Embedding(3, hidden_dim)

        # Character-level CNN
        self.char_cnn = CharCNNEncoder(num_chars=256, char_dim=64, out_dim=hidden_dim)

        # Chunk-level transformer
        self.chunk_transformer = ChunkTransformer(hidden_dim=hidden_dim)

        # Cross-resolution fusion gates
        self.gate_char = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.gate_chunk = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())

        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)

        # Scale
        self.scale = math.sqrt(hidden_dim)

    def forward(
        self,
        token_ids: torch.Tensor,
        segment_ids: Optional[torch.Tensor] = None,
        char_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len) token indices.
            segment_ids: (batch, seq_len) segment indices (0=code, 1=english, 2=reasoning).
            char_ids: (batch, seq_len, max_word_len) character indices for char CNN.

        Returns:
            (batch, seq_len, hidden_dim) fused multi-resolution embeddings.
        """
        B, L = token_ids.shape

        # Token embeddings
        tok_emb = self.token_embedding(token_ids) * self.scale  # (B, L, D)

        # Segment embeddings
        if segment_ids is not None:
            tok_emb = tok_emb + self.segment_embedding(segment_ids)

        # Character-level embeddings
        if char_ids is not None:
            char_emb = self.char_cnn(char_ids)  # (B, L, D)
        else:
            char_emb = torch.zeros_like(tok_emb)

        # Chunk-level embeddings
        chunk_emb = self.chunk_transformer(tok_emb)  # (B, L, D)

        # Cross-resolution fusion
        g_char = self.gate_char(torch.cat([tok_emb, char_emb], dim=-1))  # (B, L, D)
        g_chunk = self.gate_chunk(torch.cat([tok_emb, chunk_emb], dim=-1))  # (B, L, D)

        fused = tok_emb + g_char * char_emb + g_chunk * chunk_emb

        return self.norm(fused)


class CodeAwareRoPE(nn.Module):
    """
    Rotary Position Embeddings with code-aware offsets based on
    indentation level and bracket depth.
    """

    def __init__(self, head_dim: int = 64, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # Precompute frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

        # Code-aware offset embeddings
        self.indent_embedding = nn.Embedding(21, head_dim // 2)  # 0-20 levels
        self.bracket_embedding = nn.Embedding(11, head_dim // 2)  # 0-10 depth

    def _compute_rotary(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute sin/cos for standard RoPE."""
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))  # (L, head_dim//2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (L, head_dim)
        return emb.cos(), emb.sin()

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of x."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        indent_levels: Optional[torch.Tensor] = None,
        bracket_depths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings with optional code-aware offsets.

        Args:
            q: (batch, seq_len, num_heads, head_dim)
            k: (batch, seq_len, num_heads, head_dim)
            indent_levels: (batch, seq_len) indentation levels 0-20.
            bracket_depths: (batch, seq_len) bracket depths 0-10.

        Returns:
            Rotated (q, k) tuple.
        """
        B, L, H, D = q.shape
        cos, sin = self._compute_rotary(L, q.device)  # (L, D)
        cos = cos[None, :, None, :]  # (1, L, 1, D)
        sin = sin[None, :, None, :]  # (1, L, 1, D)

        # Code-aware angle offsets
        if indent_levels is not None and bracket_depths is not None:
            indent_offset = self.indent_embedding(indent_levels.clamp(0, 20))  # (B, L, D//2)
            bracket_offset = self.bracket_embedding(bracket_depths.clamp(0, 10))  # (B, L, D//2)
            offset = torch.cat([indent_offset, bracket_offset], dim=-1)  # (B, L, D)
            offset = offset[:, :, None, :]  # (B, L, 1, D)
            # Add offsets to the rotation angles
            cos = cos + offset.cos() * 0.1
            sin = sin + offset.sin() * 0.1

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


def detect_code_structure(
    token_ids: torch.Tensor,
    indent_token_id: int = 220,
    open_bracket_ids: Optional[list] = None,
    close_bracket_ids: Optional[list] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract indentation levels and bracket depths from token sequences.

    Args:
        token_ids: (batch, seq_len) token indices.
        indent_token_id: Token ID representing a single indent unit.
        open_bracket_ids: List of token IDs for opening brackets.
        close_bracket_ids: List of token IDs for closing brackets.

    Returns:
        Tuple of (indent_levels, bracket_depths) each (batch, seq_len).
    """
    if open_bracket_ids is None:
        open_bracket_ids = [7, 8, 9]  # Placeholder IDs for (, [, {
    if close_bracket_ids is None:
        close_bracket_ids = [10, 11, 12]  # Placeholder IDs for ), ], }

    B, L = token_ids.shape
    device = token_ids.device

    indent_levels = torch.zeros(B, L, dtype=torch.long, device=device)
    bracket_depths = torch.zeros(B, L, dtype=torch.long, device=device)

    for b in range(B):
        current_indent = 0
        current_bracket = 0
        for i in range(L):
            tid = token_ids[b, i].item()
            if tid == indent_token_id:
                current_indent += 1
            elif tid in open_bracket_ids:
                current_bracket = min(current_bracket + 1, 10)
            elif tid in close_bracket_ids:
                current_bracket = max(current_bracket - 1, 0)

            indent_levels[b, i] = min(current_indent, 20)
            bracket_depths[b, i] = current_bracket

    return indent_levels, bracket_depths
