from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


RELATIONS = ["MD", "DM", "DD"]
NEG_TAG = "random"  #负样本生成后缀
NUM_KG_RELATIONS = 6  #KG 里一共有 6 种有向关系


@dataclass
class RelationSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


@dataclass
class ScenarioSplit:
    splits: Dict[str, RelationSplit]

#先划分holdout,在做k折
def holdout_then_kfold(
    edges_dict: Dict[str, pd.DataFrame],
    test_ratio: float,
    n_folds: int,
    seed: int,
) -> Tuple[Dict[str, pd.DataFrame], List[ScenarioSplit]]:
    test_edges: Dict[str, pd.DataFrame] = {}
    dev_edges: Dict[str, pd.DataFrame] = {}

    for i, rel in enumerate(RELATIONS):
        rng = np.random.default_rng(seed + i)
        df = edges_dict[rel].reset_index(drop=True)
        idx = np.arange(len(df))
        rng.shuffle(idx)
        n_test = int(len(df) * test_ratio)
        test_edges[rel] = df.iloc[idx[:n_test]].reset_index(drop=True)
        dev_edges[rel] = df.iloc[idx[n_test:]].reset_index(drop=True)

    fold_indices: Dict[str, List[np.ndarray]] = {}
    for i, rel in enumerate(RELATIONS):
        rng = np.random.default_rng(seed + 100 + i)
        idx = np.arange(len(dev_edges[rel]))
        rng.shuffle(idx)
        fold_indices[rel] = np.array_split(idx, n_folds)

    fold_list: List[ScenarioSplit] = []
    for k in range(n_folds):
        splits: Dict[str, RelationSplit] = {}
        for rel in RELATIONS:
            folds = fold_indices[rel]
            val_idx = folds[k]
            train_parts = [folds[j] for j in range(n_folds) if j != k]
            train_idx = (
                np.concatenate(train_parts) if train_parts else np.array([], dtype=np.intp)
            )
            dev = dev_edges[rel]
            splits[rel] = RelationSplit(
                train=dev.iloc[train_idx].reset_index(drop=True),
                val=dev.iloc[val_idx].reset_index(drop=True),
                test=pd.DataFrame(columns=["h", "t"]),
            )
        fold_list.append(ScenarioSplit(splits=splits))

    return test_edges, fold_list

#判断节点类型前缀
def node_prefix(node: str) -> str:
    if node.startswith("Dr_"):
        return "Dr_"
    if node.startswith("M_"):
        return "M_"
    if node.startswith("D_"):
        return "D_"
    return ""

#6中种关系编号
_REL_TYPE_TO_INDEX = {
    ("M_", "D_"): 0,
    ("D_", "M_"): 1,
    ("Dr_", "M_"): 2,
    ("M_", "Dr_"): 3,
    ("Dr_", "D_"): 4,
    ("D_", "Dr_"): 5,
}

NODE_TYPE_TO_INDEX = {
    "M_": 0,
    "D_": 1,
    "Dr_": 2,
}
NUM_NODE_TYPES = 3


def build_node_type_tensor(
    node2id: Dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    node_type = torch.empty(len(node2id), dtype=torch.long, device=device)

    for node, idx in node2id.items():
        p = node_prefix(node)
        if p == "M_":
            node_type[idx] = 0
        elif p == "D_":
            node_type[idx] = 1
        elif p == "Dr_":
            node_type[idx] = 2
        else:
            raise ValueError(f"Unknown node type for node={node}")

    return node_type


def _rel_type_index(h: str, t: str) -> int:
    return _REL_TYPE_TO_INDEX.get((node_prefix(h), node_prefix(t)), -1)

#构建 HGT 用的 6 类关系边
def build_typed_edge_indices(
    kg_path: Path,
    node2id: Dict[str, int],
    device: torch.device,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    buckets: List[Tuple[List[int], List[int]]] = [([], []) for _ in range(NUM_KG_RELATIONS)]

    with open(kg_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            h, t = parts[0], parts[1]
            if h not in node2id or t not in node2id:
                raise KeyError(f"KG edge contains unmapped node: {h}, {t}")
            hi, ti = node2id[h], node2id[t]
            r = _rel_type_index(h, t)
            if r >= 0:
                buckets[r][0].append(hi)
                buckets[r][1].append(ti)
            r_rev = _rel_type_index(t, h)
            if r_rev >= 0:
                buckets[r_rev][0].append(ti)
                buckets[r_rev][1].append(hi)

    edge_indices = []
    for src, dst in buckets:
        if len(src) == 0:
            edge_indices.append((
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            ))
        else:
            arr = np.array(list(zip(src, dst)), dtype=np.int64)
            arr = np.unique(arr, axis=0)
            edge_indices.append((
                torch.tensor(arr[:, 0], dtype=torch.long, device=device),
                torch.tensor(arr[:, 1], dtype=torch.long, device=device),
            ))

    return edge_indices

#把正样本 DataFrame 转成数组
def df_to_edge_arr(df: pd.DataFrame, node2id: Dict[str, int]) -> np.ndarray:
    rows, cols = [], []
    for _, r in df.iterrows():
        h, t = str(r["h"]), str(r["t"])
        if h in node2id and t in node2id:
            rows.append(node2id[h])
            cols.append(node2id[t])
    if not rows:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array([rows, cols], dtype=np.int64)

#读取正样本
def load_split_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    if not {"h", "t"}.issubset(df.columns):
        raise ValueError(f"Split file must contain columns h and t: {path}")
    return df[["h", "t"]].drop_duplicates().reset_index(drop=True)

#读取一个 fold 的三任务数据
def load_scenario_from_dir(split_dir: Path, include_test: bool) -> ScenarioSplit:
    splits: Dict[str, RelationSplit] = {}
    for rel in RELATIONS:
        train = load_split_df(split_dir / f"{rel}_train.txt")
        val = load_split_df(split_dir / f"{rel}_val.txt")
        test = (
            load_split_df(split_dir / f"{rel}_test.txt")
            if include_test
            else pd.DataFrame(columns=["h", "t"])
        )
        splits[rel] = RelationSplit(train=train, val=val, test=test)
    return ScenarioSplit(splits=splits)

#读取节点映射
def load_node2id(node2id_dir: Path) -> Dict[str, int]:
    with open(node2id_dir / "node2id.json", encoding="utf-8") as f:
        node2id = json.load(f)
    return {str(k): int(v) for k, v in node2id.items()}

#读取负样本
def load_neg_arr(path: Path, node2id: Dict[str, int], expected_num: int) -> np.ndarray:
    df = pd.read_csv(path, sep="\t", dtype=str)[["h", "t"]].drop_duplicates()
    arr = np.array([
        [node2id[str(h)] for h in df["h"]],
        [node2id[str(t)] for t in df["t"]],
    ], dtype=np.int64)
    if arr.shape[1] != expected_num:
        raise RuntimeError(f"{path}: expected {expected_num}, got {arr.shape[1]}")
    return arr

#准备数据
def prepare_rel_data(
    scenario_split: ScenarioSplit,
    node2id: Dict[str, int],
    cache_dir: Path,
) -> Dict[str, Dict]:
    rel_data: Dict[str, Dict] = {}

    for rel in RELATIONS:
        split = scenario_split.splits[rel]

        tr_pos = df_to_edge_arr(split.train, node2id)
        va_pos = df_to_edge_arr(split.val, node2id)
        te_pos = df_to_edge_arr(split.test, node2id)

        tr_neg = load_neg_arr(
            cache_dir / f"{rel}_train_neg_{NEG_TAG}.txt", node2id, expected_num=tr_pos.shape[1],
        )
        va_neg = load_neg_arr(
            cache_dir / f"{rel}_val_neg_{NEG_TAG}.txt", node2id, expected_num=va_pos.shape[1],
        )

        if te_pos.shape[1] > 0:
            te_neg = load_neg_arr(
                cache_dir / f"{rel}_test_neg_{NEG_TAG}.txt", node2id, expected_num=te_pos.shape[1],
            )
        else:
            te_neg = np.zeros((2, 0), dtype=np.int64)

        rel_data[rel] = {
            "tr_pos": tr_pos, "tr_neg": tr_neg,
            "va_pos": va_pos, "va_neg": va_neg,
            "te_pos": te_pos, "te_neg": te_neg,
        }

    return rel_data
