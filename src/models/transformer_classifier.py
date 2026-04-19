from __future__ import annotations

import math

import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dk: int) -> None:
        super().__init__()
        self.dk = dk

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)

        if mask is not None:
            # mask shape is broadcastable to [B, T, T]. False entries are blocked.
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return out, attn


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        dk: int = 32,
        dv: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = dk
        self.dv = dv

        # Separate projection matrices per head.
        self.q_projs = nn.ModuleList([nn.Linear(d_model, dk) for _ in range(num_heads)])
        self.k_projs = nn.ModuleList([nn.Linear(d_model, dk) for _ in range(num_heads)])
        self.v_projs = nn.ModuleList([nn.Linear(d_model, dv) for _ in range(num_heads)])

        self.attention = ScaledDotProductAttention(dk=dk)
        self.out_proj = nn.Linear(num_heads * dv, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, D]
        head_outputs: list[torch.Tensor] = []
        head_weights: list[torch.Tensor] = []

        attn_mask = None
        if padding_mask is not None:
            # Convert [B, T] -> [B, T, T]
            attn_mask = padding_mask.unsqueeze(1) & padding_mask.unsqueeze(2)

        for q_proj, k_proj, v_proj in zip(self.q_projs, self.k_projs, self.v_projs):
            q = q_proj(x)
            k = k_proj(x)
            v = v_proj(x)
            out, weights = self.attention(q, k, v, mask=attn_mask)
            head_outputs.append(out)
            head_weights.append(weights)

        multi = torch.cat(head_outputs, dim=-1)
        projected = self.out_proj(multi)
        projected = self.dropout(projected)

        # [B, H, T, T]
        stacked_weights = torch.stack(head_weights, dim=1)
        return projected, stacked_weights


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int = 128, d_ff: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 128, max_len: int = 2048) -> None:
        super().__init__()

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0)


class PreLNEncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        dk: int = 32,
        dv: int = 32,
        d_ff: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.self_attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dk=dk,
            dv=dv,
            dropout=dropout,
        )
        self.ffn = PositionwiseFFN(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_in = self.ln1(x)
        attn_out, attn_weights = self.self_attn(attn_in, padding_mask=padding_mask)
        x = x + self.dropout(attn_out)

        ffn_in = self.ln2(x)
        ffn_out = self.ffn(ffn_in)
        x = x + self.dropout(ffn_out)
        return x, attn_weights


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 4,
        d_model: int = 128,
        num_heads: int = 4,
        dk: int = 32,
        dv: int = 32,
        d_ff: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                PreLNEncoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    dk=dk,
                    dv=dv,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        return_all_attention: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        all_attn: list[torch.Tensor] = []
        for layer in self.layers:
            x, attn = layer(x, padding_mask=padding_mask)
            if return_all_attention:
                all_attn.append(attn)
        return x, all_attn


class ClassificationHead(nn.Module):
    def __init__(self, d_model: int = 128, hidden_dim: int = 64, num_classes: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, cls_repr: torch.Tensor) -> torch.Tensor:
        return self.net(cls_repr)


class TransformerTopicClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 5,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dk: int = 32,
        dv: int = 32,
        d_ff: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
        max_len: int = 257,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.positional = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)

        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dk=dk,
            dv=dv,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.classifier = ClassificationHead(d_model=d_model, hidden_dim=64, num_classes=num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]] | torch.Tensor:
        batch_size = input_ids.size(0)

        x = self.token_embedding(input_ids)
        cls = self.cls_token.expand(batch_size, 1, -1)
        x = torch.cat([cls, x], dim=1)

        cls_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=input_ids.device)
        full_mask = torch.cat([cls_mask, attention_mask], dim=1)

        x = self.positional(x)
        enc_out, all_attn = self.encoder(
            x,
            padding_mask=full_mask,
            return_all_attention=return_attention,
        )
        cls_out = enc_out[:, 0]
        logits = self.classifier(cls_out)

        if return_attention:
            return logits, all_attn
        return logits
