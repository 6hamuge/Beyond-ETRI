"""PTDTransformer: a deviation-aware Transformer for multi-task sleep prediction.

Input ``x`` of shape ``(batch, seq_len, n_features)`` is split at the midpoint
into a raw-behavior block and a personal-deviation block. ``DeviationAwareAttention``
fuses them (deviation as query, raw as key/value); a CLS-token Transformer
encoder pools the sequence; a shared MLP feeds 7 per-label heads.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 365, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class DeviationAwareAttention(nn.Module):
    """Cross-attention fusion of the raw and deviation feature branches."""

    def __init__(
        self, d_model: int, n_heads: int, n_raw_feat: int, n_dev_feat: int
    ) -> None:
        super().__init__()
        self.raw_proj = nn.Linear(n_raw_feat, d_model)
        self.dev_proj = nn.Linear(n_dev_feat, d_model)
        self.fusion = nn.Linear(2 * d_model, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, raw_feat: torch.Tensor, dev_feat: torch.Tensor) -> torch.Tensor:
        raw_emb = F.gelu(self.raw_proj(raw_feat))
        dev_emb = F.gelu(self.dev_proj(dev_feat))
        fused = self.fusion(torch.cat([raw_emb, dev_emb], dim=-1))
        attn_out, _ = self.attn(dev_emb, raw_emb, raw_emb)
        return self.norm(fused + attn_out)


class PTDTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_labels: int = 7,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        seq_len: int = 7,
    ) -> None:
        super().__init__()

        # Input is [raw block || deviation block]; the two halves are equal size.
        self.n_raw = n_features // 2
        self.n_dev = n_features - self.n_raw

        self.dev_attn = DeviationAwareAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_raw_feat=self.n_raw,
            n_dev_feat=self.n_dev,
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_enc = PositionalEncoding(
            d_model=d_model, max_len=seq_len + 1, dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.shared_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        self.task_heads = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, 1))
                for _ in range(n_labels)
            ]
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        raw = x[:, :, : self.n_raw]
        dev = x[:, :, self.n_raw:]

        emb = self.dev_attn(raw, dev)
        cls = self.cls_token.expand(batch_size, -1, -1)
        emb = torch.cat([cls, emb], dim=1)
        emb = self.pos_enc(emb)

        enc = self.transformer(emb)
        cls_out = enc[:, 0]
        shared = self.shared_head(cls_out)

        return torch.cat([head(shared) for head in self.task_heads], dim=-1)
