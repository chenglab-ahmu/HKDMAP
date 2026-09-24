from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


from src.hgt import HGTEncoder
from src.gat import GAT


#融合模块
class DualChannelFusion(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, z_struct: torch.Tensor, z_bio: torch.Tensor) -> torch.Tensor:
        return self.norm(0.5 * z_struct + 0.5 * z_bio)


class DualViewModel(nn.Module):

    def __init__(
            self,
            mp2v_dim: int,
            tax_dim: int,
            mesh_dim: int,
            drug_dim: int,
            hidden_dim: int = 128,
            emb_dim: int = 64,
            num_node_types: int = 3,
            num_kg_rels: int = 6,
            num_heads: int = 4,
            struct_layers: int = 3,
            gat_layers: int = 2,
            dropout: float = 0.2,
            bio_kg_layers: int = 0,
    ):
        super().__init__()
        self.bio_kg_layers = int(bio_kg_layers)
        self._emb_dim = emb_dim

        self.hgt_struct = HGTEncoder(
            in_dim=mp2v_dim,
            hidden_dim=hidden_dim,
            out_dim=emb_dim,
            num_node_types=num_node_types,
            num_relations=num_kg_rels,
            num_heads=num_heads,
            num_layers=struct_layers,
            dropout=dropout,
        )

        self.gat_M = GAT(tax_dim, hidden_dim, emb_dim, num_heads, gat_layers, dropout)
        self.gat_D = GAT(mesh_dim, hidden_dim, emb_dim, num_heads, gat_layers, dropout)
        self.gat_Dr = GAT(drug_dim, hidden_dim, emb_dim, num_heads, gat_layers, dropout)

        self.hgt_bio = HGTEncoder(
            in_dim=emb_dim,
            hidden_dim=hidden_dim,
            out_dim=emb_dim,
            num_node_types=num_node_types,
            num_relations=num_kg_rels,
            num_heads=num_heads,
            num_layers=self.bio_kg_layers,
            dropout=dropout,
        ) if self.bio_kg_layers > 0 else None

        self.fusion = DualChannelFusion(emb_dim)

    @property
    def output_dim(self) -> int:
        return self._emb_dim

    def _compute_bio(
            self,
            ext_M: torch.Tensor,
            ext_D: torch.Tensor,
            ext_Dr: torch.Tensor,
            kg_edges: List[Tuple[torch.Tensor, torch.Tensor]],
            ei_mm: Tuple[torch.Tensor, torch.Tensor],
            ei_dd: Tuple[torch.Tensor, torch.Tensor],
            ei_drdr: Tuple[torch.Tensor, torch.Tensor],
            m_idx: torch.Tensor,
            d_idx: torch.Tensor,
            dr_idx: torch.Tensor,
            N: int,
            device: torch.device,
            dtype: torch.dtype,
            node_type: torch.Tensor,
    ) -> torch.Tensor:
        z_bio = torch.zeros(N, self._emb_dim, device=device, dtype=dtype)
        z_bio[m_idx] = self.gat_M(ext_M, ei_mm)
        z_bio[d_idx] = self.gat_D(ext_D, ei_dd)
        z_bio[dr_idx] = self.gat_Dr(ext_Dr, ei_drdr)

        if self.hgt_bio is not None:
            z_bio = self.hgt_bio(z_bio, kg_edges, node_type)

        return z_bio

    def forward(
            self,
            mp2v_feat: torch.Tensor,
            ext_M: torch.Tensor,
            ext_D: torch.Tensor,
            ext_Dr: torch.Tensor,
            kg_edges: List[Tuple[torch.Tensor, torch.Tensor]],
            sim_mm: Tuple[torch.Tensor, torch.Tensor],
            sim_dd: Tuple[torch.Tensor, torch.Tensor],
            sim_drdr: Tuple[torch.Tensor, torch.Tensor],
            m_idx: torch.Tensor,
            d_idx: torch.Tensor,
            dr_idx: torch.Tensor,
            node_type: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = mp2v_feat.size(0)
        device, dtype = mp2v_feat.device, mp2v_feat.dtype

        z_struct = self.hgt_struct(mp2v_feat, kg_edges, node_type)
        z_bio = self._compute_bio(
            ext_M, ext_D, ext_Dr, kg_edges,
            sim_mm, sim_dd, sim_drdr,
            m_idx, d_idx, dr_idx, N, device, dtype,
            node_type,
        )
        return self.fusion(z_struct, z_bio), z_struct, z_bio
