from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.2,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        assert out_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.out_dim = out_dim
        self.dropout = dropout
        self.negative_slope = negative_slope

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Parameter(torch.empty(num_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.attn)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        src, dst = edge_index
        N, H, D = h.size(0), self.num_heads, self.head_dim

        Wh = self.W(h).view(N, H, D)  # (N, H, D)

        # Attention logits
        e = F.leaky_relu(
            (torch.cat([Wh[dst], Wh[src]], dim=-1) * self.attn).sum(-1),
            negative_slope=self.negative_slope,
        )  # (E, H)

        alpha = self._edge_softmax(e, dst, N)               # (E, H)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Weighted neighborhood aggregation
        msg = alpha.unsqueeze(-1) * Wh[src]                 # (E, H, D)
        agg = torch.zeros(N, H, D, device=h.device, dtype=h.dtype)
        for hi in range(H):
            agg[:, hi].index_add_(0, dst, msg[:, hi])

        out = self.norm((Wh + agg).reshape(N, self.out_dim))
        return F.dropout(out, p=self.dropout, training=self.training)

    def _edge_softmax(self, e: torch.Tensor, dst: torch.Tensor, N: int) -> torch.Tensor:
        alpha = torch.empty_like(e)
        for hi in range(e.size(1)):
            e_hi = e[:, hi]
            e_max = torch.full((N,), -1e9, device=e.device)
            e_max.scatter_reduce_(0, dst, e_hi, reduce="amax", include_self=True)
            exp_e = torch.exp(e_hi - e_max[dst])
            sum_exp = torch.zeros(N, device=e.device)
            sum_exp.index_add_(0, dst, exp_e)
            alpha[:, hi] = exp_e / (sum_exp[dst] + 1e-12)
        return alpha


class GAT(nn.Module):

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([
            GATLayer(
                dims[i], dims[i + 1],
                num_heads=num_heads,
                dropout=dropout if i < num_layers - 1 else 0.0,
            )
            for i in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index)
            if i < self.num_layers - 1:
                h = F.elu(h)
        return h
