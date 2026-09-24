from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HGTLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_node_types: int,
        num_relations: int,
        num_heads: int = 4,
        dropout: float = 0.2,
        use_norm: bool = True,
        cross_reducer: str = "mean",
    ):
        super().__init__()

        if out_dim % num_heads != 0:
            raise ValueError(
                f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})"
            )
        if num_node_types < 1:
            raise ValueError(f"num_node_types must be >= 1, got {num_node_types}")
        if num_relations < 1:
            raise ValueError(f"num_relations must be >= 1, got {num_relations}")
        if cross_reducer not in {"mean", "sum"}:
            raise ValueError(f"Unsupported cross_reducer={cross_reducer}")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_node_types = int(num_node_types)
        self.num_relations = int(num_relations)
        self.num_heads = int(num_heads)
        self.head_dim = out_dim // num_heads
        self.sqrt_dk = math.sqrt(self.head_dim)
        self.cross_reducer = cross_reducer
        self.use_norm = use_norm


        # M / D / Dr 三类节点各自拥有自己的 K/Q/V/A 映射
        self.k_linears = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for _ in range(num_node_types)
        ])
        self.q_linears = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for _ in range(num_node_types)
        ])
        self.v_linears = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for _ in range(num_node_types)
        ])
        self.a_linears = nn.ModuleList([
            nn.Linear(out_dim, out_dim) for _ in range(num_node_types)
        ])


        # relation_att: K_source 经边类型矩阵变换后再和 Q_target 点积
        # relation_msg: V_source 经边类型矩阵变换后作为消息
        # relation_pri: 每种关系、每个 head 的先验权重
        self.relation_att = nn.Parameter(
            torch.empty(num_relations, num_heads, self.head_dim, self.head_dim)
        )
        self.relation_msg = nn.Parameter(
            torch.empty(num_relations, num_heads, self.head_dim, self.head_dim)
        )
        self.relation_pri = nn.Parameter(torch.ones(num_relations, num_heads))


        # sigmoid(skip[t]) 控制新消息和残差之间的融合比例
        self.skip = nn.Parameter(torch.ones(num_node_types))

        # 如果输入输出维度不同，需要把 residual 投影到 out_dim
        self.residual_proj = (
            None
            if in_dim == out_dim
            else nn.ModuleList([
                nn.Linear(in_dim, out_dim, bias=False)
                for _ in range(num_node_types)
            ])
        )

        self.drop = nn.Dropout(dropout)

        self.norms = (
            nn.ModuleList([nn.LayerNorm(out_dim) for _ in range(num_node_types)])
            if use_norm
            else None
        )

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(
        self,
        h: torch.Tensor,
        edge_indices: List[Tuple[torch.Tensor, torch.Tensor]],
        node_type: torch.Tensor,
    ) -> torch.Tensor:
        if len(edge_indices) != self.num_relations:
            raise ValueError(
                f"Expected {self.num_relations} relation edge lists, "
                f"got {len(edge_indices)}"
            )
        if node_type.shape[0] != h.shape[0]:
            raise ValueError(
                f"node_type length {node_type.shape[0]} does not match "
                f"number of nodes {h.shape[0]}"
            )
        if int(node_type.min()) < 0 or int(node_type.max()) >= self.num_node_types:
            raise ValueError(
                f"node_type values must be in [0, {self.num_node_types - 1}]"
            )

        N = h.size(0)
        device, dtype = h.device, h.dtype
        H, D = self.num_heads, self.head_dim

        # Step 1: 按节点类型计算 Q/K/V
        q_all = torch.zeros(N, H, D, device=device, dtype=dtype)
        k_all = torch.zeros(N, H, D, device=device, dtype=dtype)
        v_all = torch.zeros(N, H, D, device=device, dtype=dtype)

        for t in range(self.num_node_types):
            mask = node_type == t
            if not torch.any(mask):
                continue

            q_all[mask] = self.q_linears[t](h[mask]).view(-1, H, D)
            k_all[mask] = self.k_linears[t](h[mask]).view(-1, H, D)
            v_all[mask] = self.v_linears[t](h[mask]).view(-1, H, D)

        # Step 2: 每种关系单独计算 HGT attention/message
        rel_outs: List[torch.Tensor] = []
        rel_masks: List[torch.Tensor] = []

        for r, (src, dst) in enumerate(edge_indices):
            if src.numel() == 0:
                continue

            k_src = k_all[src]      # [E, H, D]
            q_dst = q_all[dst]      # [E, H, D]
            v_src = v_all[src]      # [E, H, D]

            # K_source * W_ATT_relation
            rel_att = self.relation_att[r]  # [H, D, D]
            key = torch.einsum("ehd,hdf->ehf", k_src, rel_att)

            # attention score:
            #   Q_target dot (K_source W_ATT_relation)
            #   * relation_prior / sqrt(d)
            score = (q_dst * key).sum(dim=-1)
            score = score * self.relation_pri[r] / self.sqrt_dk

            alpha = self._multi_head_edge_softmax(score, dst, N)
            alpha = self.drop(alpha)

            # message:
            #   V_source * W_MSG_relation
            rel_msg = self.relation_msg[r]  # [H, D, D]
            val = torch.einsum("ehd,hdf->ehf", v_src, rel_msg)

            msg = alpha.unsqueeze(-1) * val
            msg = msg.reshape(-1, self.out_dim)

            out_r = torch.zeros(N, self.out_dim, device=device, dtype=dtype)
            out_r.index_add_(0, dst, msg)

            rel_outs.append(out_r)

            # 记录每个节点是否收到该 relation 的消息，
            # 用于跨 relation 做 mean reducer。
            has_msg = torch.zeros(N, 1, device=device, dtype=dtype)
            has_msg[dst.unique()] = 1.0
            rel_masks.append(has_msg)

        # Step 3: 跨关系聚合
        if not rel_outs:
            agg = torch.zeros(N, self.out_dim, device=device, dtype=dtype)
        else:
            stacked = torch.stack(rel_outs, dim=0)   # [R_active, N, out_dim]
            masks = torch.stack(rel_masks, dim=0)    # [R_active, N, 1]

            if self.cross_reducer == "mean":
                agg = stacked.sum(dim=0) / masks.sum(dim=0).clamp_min(1.0)
            elif self.cross_reducer == "sum":
                agg = stacked.sum(dim=0)
            else:
                raise ValueError(f"Unsupported cross_reducer={self.cross_reducer}")


        # Step 4: 按目标节点类型做 A-Linear + skip + norm
        out = torch.zeros(N, self.out_dim, device=device, dtype=dtype)

        for t in range(self.num_node_types):
            mask = node_type == t
            if not torch.any(mask):
                continue

            trans = self.a_linears[t](agg[mask])
            alpha_skip = torch.sigmoid(self.skip[t])

            if self.residual_proj is None:
                residual = h[mask]
            else:
                residual = self.residual_proj[t](h[mask])

            # 新消息与残差通过 type-specific skip gate 融合
            trans = trans * alpha_skip + residual * (1.0 - alpha_skip)

            if self.use_norm:
                trans = self.norms[t](trans)

            out[mask] = self.drop(trans)

        return out

    def _multi_head_edge_softmax(
        self,
        e: torch.Tensor,
        dst: torch.Tensor,
        N: int,
    ) -> torch.Tensor:
        alpha = torch.empty_like(e)
        for hi in range(e.size(1)):
            alpha[:, hi] = self._edge_softmax(e[:, hi], dst, N)
        return alpha

    @staticmethod
    def _edge_softmax(e: torch.Tensor, dst: torch.Tensor, N: int) -> torch.Tensor:
        e_max = torch.full((N,), -1e9, device=e.device, dtype=e.dtype)
        e_max.scatter_reduce_(0, dst, e, reduce="amax", include_self=True)

        exp_e = torch.exp(e - e_max[dst])
        sum_exp = torch.zeros(N, device=e.device, dtype=e.dtype)
        sum_exp.index_add_(0, dst, exp_e)

        return exp_e / (sum_exp[dst] + 1e-12)


class HGTEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_node_types: int,
        num_relations: int,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
        use_norm: bool = True,
        cross_reducer: str = "mean",
    ):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]

        self.layers = nn.ModuleList([
            HGTLayer(
                in_dim=dims[i],
                out_dim=dims[i + 1],
                num_node_types=num_node_types,
                num_relations=num_relations,
                num_heads=num_heads,
                dropout=dropout if i < num_layers - 1 else 0.0,
                use_norm=use_norm,
                cross_reducer=cross_reducer,
            )
            for i in range(num_layers)
        ])

        self.num_layers = num_layers

    def forward(
        self,
        x: torch.Tensor,
        edge_indices: List[Tuple[torch.Tensor, torch.Tensor]],
        node_type: torch.Tensor,
    ) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_indices, node_type)
            if i < self.num_layers - 1:
                h = F.elu(h)
        return h