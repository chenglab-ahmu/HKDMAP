from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data_utils import RELATIONS

#给一对节点 (src, dst) 打分，输出一个 logit
class MLPDecoder(nn.Module):
    def __init__(self, emb_dim: int, dropout: float = 0.1):   #MLP的dropout固定写成0.1---暂时没有调整这个参数
        super().__init__()
        hidden = emb_dim * 2
        self.heads = nn.ModuleDict({
            r: nn.Sequential(
                nn.Linear(emb_dim * 2, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
            for r in RELATIONS
        })
        for r in RELATIONS:
            nn.init.xavier_uniform_(self.heads[r][-1].weight, gain=0.01)
            nn.init.zeros_(self.heads[r][-1].bias)

    def forward(
        self,
        z: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        relation: str,
    ) -> torch.Tensor:
        pair = torch.cat([z[src_idx], z[dst_idx]], dim=-1)
        return self.heads[relation](pair).squeeze(-1)

#自适应多任务 loss 平衡
class UncertaintyWeighting(nn.Module):
    def __init__(self, task_names: List[str]):
        super().__init__()
        self.log_vars = nn.ParameterDict()
        for name in task_names:
            self.log_vars[name] = nn.Parameter(torch.zeros(1))  #给每个任务创建一个可以学习的参数，初始都是0

    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = torch.tensor(0.0, device=next(iter(losses.values())).device)
        for name, loss in losses.items():
            log_var = self.log_vars[name]
            precision = torch.exp(-log_var)
            total = total + 0.5 * (precision * loss + log_var)  #不确定性加权公式
        return total

#计算三任务loss
def compute_multitask_loss(
    z: torch.Tensor,
    decoder: nn.Module,
    rel_data: Dict[str, Dict],
    device: torch.device,
    label_smooth: float,
    uw: UncertaintyWeighting | None = None,
) -> torch.Tensor:
    per_task: Dict[str, torch.Tensor] = {}

    for rel in RELATIONS:
        rd = rel_data[rel]
        if rd["tr_pos"].shape[1] == 0:
            continue

        pos_s = torch.from_numpy(rd["tr_pos"][0]).long().to(device)
        pos_d = torch.from_numpy(rd["tr_pos"][1]).long().to(device)
        neg_s = torch.from_numpy(rd["tr_neg"][0]).long().to(device)
        neg_d = torch.from_numpy(rd["tr_neg"][1]).long().to(device)

        pos_logits = decoder(z, pos_s, pos_d, rel)
        neg_logits = decoder(z, neg_s, neg_d, rel)

        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([
            torch.full_like(pos_logits, 1.0 - label_smooth),
            torch.full_like(neg_logits, label_smooth),
        ])
        per_task[rel] = F.binary_cross_entropy_with_logits(logits, labels) #计算当前任务的BCEloss

    if not per_task:
        return torch.tensor(0.0, device=device)

    if uw is not None:
        return uw(per_task)  #如果不使用自适应加权，就用mean，三个loss平均，再配置文件中可以改

    return sum(per_task.values()) / len(per_task)
