from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.config_utils import add_config_arg, load_config, ensure_dir
from src.data_utils import RELATIONS, NEG_TAG, ScenarioSplit, RelationSplit, holdout_then_kfold

_REL_SEED_OFFSET = {"MD": 0, "DM": 10000, "DD": 20000}    #给不同的关系设置不同随机种子偏置，三种关系的负样本采样随机流不同，但整体仍然可复现
_NUM_DOT0_RE = re.compile(r"^(\d+)\.0$")
_WS_RE = re.compile(r"\s+")

#原始ID标准化
def canonicalize_token(x: object) -> Optional[str]:
    if x is None:
        return None

    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None

    m = _NUM_DOT0_RE.match(s)
    if m:
        s = m.group(1)

    s = _WS_RE.sub(" ", s)
    s = s.replace(" ", "__")
    return s.lower()

#读取两列表格并作标准化
def read_two_col_tsv(path: str | Path) -> List[Tuple[str, str]]:
    path = Path(path)
    pairs: List[Tuple[str, str]] = []

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        first = True

        for row in reader:
            if not row:
                continue

            if first:
                first = False
                low = [c.strip().lower() for c in row[:2]]
                if low in (
                    ["microbe", "disease"],
                    ["drug", "microbe"],
                    ["drug", "disease"],
                ):
                    continue

            if len(row) < 2:
                continue

            h = canonicalize_token(row[0])
            t = canonicalize_token(row[1])
            if h is not None and t is not None:
                pairs.append((h, t))

    return pairs

#把边列表协会两列TSV-注意写回的是没有表头的
def write_two_col_tsv(path: str | Path, pairs: Iterable[Tuple[str, str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for h, t in pairs:
            writer.writerow([h, t])

#由于DD关联数据太多，固对DD数据进行过滤，过滤的条件是drug必须都出现在微生物和疾病的关联数据里
def filter_dd(
    dd_raw: List[Tuple[str, str]],
    md_raw: List[Tuple[str, str]],
    dm_raw: List[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    dm_drugs = {dr for dr, _ in dm_raw}
    md_diseases = {d for _, d in md_raw}
    return [(dr, d) for dr, d in dd_raw if dr in dm_drugs and d in md_diseases]

#读取 raw 三类原始边--标准化--过滤DD--写入processed三类文件，但会每类关系的数量统计
def build_processed_files(raw_dir: str | Path, out_dir: str | Path) -> Dict[str, int]:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_raw = read_two_col_tsv(raw_dir / "microbe-disease.txt")
    dm_raw = read_two_col_tsv(raw_dir / "drug-microbe.txt")
    dd_raw = read_two_col_tsv(raw_dir / "drug-disease.txt")
    dd_filt = filter_dd(dd_raw, md_raw, dm_raw)

    write_two_col_tsv(out_dir / "microbe_disease.txt", md_raw)
    write_two_col_tsv(out_dir / "drug_microbe.txt", dm_raw)
    write_two_col_tsv(out_dir / "drug_disease_filt.txt", dd_filt)

    return {
        "MD": len(md_raw),
        "DM": len(dm_raw),
        "DD_raw": len(dd_raw),
        "DD_filtered": len(dd_filt),
    }

#读取processed目录下无表头的两列文件
def _read_processed_two_col(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, dtype=str, names=["h", "t"])
    return df[["h", "t"]].dropna().drop_duplicates().reset_index(drop=True)

#这个函数从 processed 文件中读取三类关系，并给节点加前缀：微生物：M_，疾病：D_，药物：Dr_
def load_all_relation_edges(processed_dir: str | Path) -> Dict[str, pd.DataFrame]:
    processed_dir = Path(processed_dir)

    md = _read_processed_two_col(processed_dir / "microbe_disease.txt")
    dm = _read_processed_two_col(processed_dir / "drug_microbe.txt")
    dd = _read_processed_two_col(processed_dir / "drug_disease_filt.txt")

    md = pd.DataFrame({
        "h": md["h"].astype(str).map(lambda x: f"M_{x}"),
        "t": md["t"].astype(str).map(lambda x: f"D_{x}"),
    }).drop_duplicates().reset_index(drop=True)

    dm = pd.DataFrame({
        "h": dm["h"].astype(str).map(lambda x: f"Dr_{x}"),
        "t": dm["t"].astype(str).map(lambda x: f"M_{x}"),
    }).drop_duplicates().reset_index(drop=True)

    dd = pd.DataFrame({
        "h": dd["h"].astype(str).map(lambda x: f"Dr_{x}"),
        "t": dd["t"].astype(str).map(lambda x: f"D_{x}"),
    }).drop_duplicates().reset_index(drop=True)

    return {"MD": md, "DM": dm, "DD": dd}

# 根据某个 fold/final 目录中的训练正样本，构造 train-only KG
def build_train_only_kg_from_split_dir(
    split_dir: str | Path,
    relations: Iterable[str] = ("MD", "DM", "DD"),
) -> Path:
    split_dir = Path(split_dir)
    kg_out = split_dir / "kg.edgelist"
    kg_out.parent.mkdir(parents=True, exist_ok=True)

    edge_set: set[tuple[str, str]] = set()

    for rel in relations:
        train_path = split_dir / f"{rel}_train.txt"
        df = pd.read_csv(train_path, sep="\t", dtype=str)
        df = df[["h", "t"]].dropna().drop_duplicates().reset_index(drop=True)

        for _, row in df.iterrows():
            edge_set.add((str(row["h"]), str(row["t"])))

    with open(kg_out, "w", encoding="utf-8") as f:
        for h, t in sorted(edge_set):
            f.write(f"{h}\t{t}\n")

    return kg_out



# Positive/negative split preparation
def _save_edge_df(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[["h", "t"]].astype(str).drop_duplicates().reset_index(drop=True)
    out.to_csv(path, sep="\t", index=False)
    return out

#把 DataFrame 边表转成集合
def _df_to_name_set(df: pd.DataFrame) -> set[tuple[str, str]]:
    if df is None or len(df) == 0:
        return set()

    x = df[["h", "t"]].astype(str).drop_duplicates()
    return set(zip(x["h"].tolist(), x["t"].tolist()))

#保存全局节点映射表
def _write_node2id(node2id: Dict[str, int], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "node2id.json", "w", encoding="utf-8") as f:
        json.dump(node2id, f, ensure_ascii=False, indent=2)

#从三类正样本边中收集所有节点，并生成全局 node2id
def _collect_global_nodes_from_edges(edges_dict: Dict[str, pd.DataFrame]) -> Dict[str, int]:
    nodes: set[str] = set()    #用集合保存所有节点，自动去重。

    for rel in RELATIONS:
        df = edges_dict[rel]
        nodes.update(df["h"].astype(str).tolist())
        nodes.update(df["t"].astype(str).tolist())

    return {n: i for i, n in enumerate(sorted(nodes))}   #对节点排序后编号

#对某一个关系，先构造全局候选负样本池，然后无放回采样出和正样本数量相同的负样本
def _sample_fixed_global_negatives_for_relation(
    *,
    rel: str,
    all_pos_edges_dict: Dict[str, pd.DataFrame],
    seed: int,
) -> pd.DataFrame:
    pos_df = (
        all_pos_edges_dict[rel][["h", "t"]]
        .astype(str)
        .drop_duplicates()
        .reset_index(drop=True)
    )     #取当前关系的全部正样本，去重
    p_all = _df_to_name_set(pos_df)   #把全部正样本转成集合

    src_names = sorted(pos_df["h"].unique().tolist())
    dst_names = sorted(pos_df["t"].unique().tolist())    #取当前关系里出现过的所有源节点和目标节点

    candidates = [
        (h, t)
        for h in src_names
        for t in dst_names
        if (h, t) not in p_all
    ]    #构造负样本候选池：源节点集合 × 目标节点集合 - 已知正样本集合

    required = len(pos_df)   #负样本数量设置为正样本数量
    if len(candidates) < required:
        raise RuntimeError(f"[{rel}] insufficient negatives: available={len(candidates)}, required={required}")  #果候选负样本数量不够，就报错

    rng = np.random.default_rng(seed + _REL_SEED_OFFSET.get(rel, 0))           #为当前关系创建随机数生成器，不同关系用不同 seed 偏移
    chosen_idx = rng.choice(len(candidates), size=required, replace=False)     #从候选负样本中无放回采样
    selected = [candidates[int(i)] for i in chosen_idx]                        #根据采样索引取出负样本边

    return pd.DataFrame(selected, columns=["h", "t"]).sort_values(["h", "t"]).reset_index(drop=True)  #返回负样本 DataFrame，并排序

#构造 final 训练场景
def _make_final_scenario(
    fold_list: list[ScenarioSplit],
    test_edges: Dict[str, pd.DataFrame],
) -> ScenarioSplit:
    final_splits: Dict[str, RelationSplit] = {}   ##准备保存三类关系的 final split

    for rel in RELATIONS:
        s0 = fold_list[0].splits[rel]   #fold_0.train + fold_0.val 正好等于完整 dev 数据
        dev = (
            pd.concat([s0.train, s0.val], ignore_index=True)
            .drop_duplicates()
            .reset_index(drop=True)
        )   #把第 0 折的 train 和 val 合并成完整 dev
        final_splits[rel] = RelationSplit(train=dev, val=dev, test=test_edges[rel])   #finall阶段：val=dev

    return ScenarioSplit(splits=final_splits)

#检查某个 split 的正负样本数量是否一致
def _assert_same_count(
    rel: str,
    split_name: str,
    pos_set: set[tuple[str, str]],
    neg_set: set[tuple[str, str]],
) -> None:
    if len(pos_set) != len(neg_set):
        raise RuntimeError(f"[{rel}] {split_name} pos/neg count mismatch: {len(pos_set)} vs {len(neg_set)}")   #如果数量不一致，就报错

#这个函数负责把一个场景保存到目录中
def _save_prepared_scenario(
    *,
    pos_scenario: ScenarioSplit,  #正样本划分
    neg_scenario: ScenarioSplit,   #负样本划分
    cache_dir: Path,         #保存目录
    include_test: bool,     #是否保存 test
    allow_train_val_overlap: bool = False,   #是否允许 train_neg 和 val_neg 重叠
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)   #确保保存目录存在

    for rel in RELATIONS:
        p = pos_scenario.splits[rel]
        n = neg_scenario.splits[rel]    #取当前关系的正样本和负样本划分

        pos_train = _save_edge_df(p.train, cache_dir / f"{rel}_train.txt")
        pos_val = _save_edge_df(p.val, cache_dir / f"{rel}_val.txt")    #保存正样本 train / val
        neg_train = _save_edge_df(n.train, cache_dir / f"{rel}_train_neg_{NEG_TAG}.txt")
        neg_val = _save_edge_df(n.val, cache_dir / f"{rel}_val_neg_{NEG_TAG}.txt")   #保存负样本 train / val

        pos_train_set = _df_to_name_set(pos_train)
        pos_val_set = _df_to_name_set(pos_val)
        neg_train_set = _df_to_name_set(neg_train)
        neg_val_set = _df_to_name_set(neg_val)    #转成集合，用于检查数量和重叠

        _assert_same_count(rel, "train", pos_train_set, neg_train_set)
        _assert_same_count(rel, "val", pos_val_set, neg_val_set)   #检查 train 和 val 的正负样本数量一致

        pos_all = pos_train_set | pos_val_set
        neg_all = neg_train_set | neg_val_set    #合并 train/val 的正样本集合和负样本集合

        if include_test:
            pos_test = _save_edge_df(p.test, cache_dir / f"{rel}_test.txt")
            neg_test = _save_edge_df(n.test, cache_dir / f"{rel}_test_neg_{NEG_TAG}.txt")  #保存 test 正负样本

            pos_test_set = _df_to_name_set(pos_test)
            neg_test_set = _df_to_name_set(neg_test)   #转成集合

            _assert_same_count(rel, "test", pos_test_set, neg_test_set)  #查 test 正负样本数量一致

            pos_all |= pos_test_set
            neg_all |= neg_test_set    #把 test 也加入全局检查集合

            if neg_train_set & neg_test_set:
                raise RuntimeError(f"[{rel}] train_neg overlaps test_neg")
            if neg_val_set & neg_test_set:
                raise RuntimeError(f"[{rel}] val_neg overlaps test_neg")   #检查负样本之间是否重叠

        if pos_all & neg_all:
            raise RuntimeError(f"[{rel}] scenario pos/neg overlap")   #检查正样本和负样本是否有重叠

        if not allow_train_val_overlap and (neg_train_set & neg_val_set):
            raise RuntimeError(f"[{rel}] train_neg overlaps val_neg")

        if include_test:
            print(
                f"  [{rel}] pos train/val/test="
                f"{len(pos_train_set)}/{len(pos_val_set)}/{len(pos_test_set)}  "
                f"neg train/val/test="
                f"{len(neg_train_set)}/{len(neg_val_set)}/{len(neg_test_set)}"
            )
        else:
            print(
                f"  [{rel}] pos train/val="
                f"{len(pos_train_set)}/{len(pos_val_set)}  "
                f"neg train/val="
                f"{len(neg_train_set)}/{len(neg_val_set)}"
            )

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prepare processed relations, positive/negative splits, and train-only KGs.",
    )   #创建命令行参数解析器
    add_config_arg(ap)
    ap.add_argument("--cv", action="store_true", help="Generate holdout + K-fold prepared samples.")
    args = ap.parse_args()

    base = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config)
    seed = int(cfg["seed"])

    random.seed(seed)
    np.random.seed(seed)    #固定 Python 和 NumPy 的随机种子

    raw_dir = base / cfg["paths"]["raw_dir"]
    proc_dir = ensure_dir(base / cfg["paths"]["processed_dir"])    #从配置读取 raw 和 processed 目录

    print(f"[PrepareDataset] config={args.config}  cv={args.cv}  neg={NEG_TAG}")

    stats = build_processed_files(raw_dir=raw_dir, out_dir=proc_dir)   #执行 raw → processed
    print(
        f"[Processed] MD={stats['MD']}  DM={stats['DM']}  "
        f"DD={stats['DD_filtered']}/{stats['DD_raw']}"     #打印 processed 统计
    )

    edges_dict = load_all_relation_edges(proc_dir)

    node2id_global = _collect_global_nodes_from_edges(edges_dict)      #从三类正样本边中收集所有节点，生成全局 node2id
    pos_summary = "  ".join(f"{rel}={len(edges_dict[rel])}" for rel in RELATIONS)
    print(f"[P_all] {pos_summary}  nodes={len(node2id_global)}")   #打印全局正样本数量和节点数量

    neg_edges_dict: Dict[str, pd.DataFrame] = {
        rel: _sample_fixed_global_negatives_for_relation(
            rel=rel,
            all_pos_edges_dict=edges_dict,
            seed=seed,
        )
        for rel in RELATIONS
    }   #为每个关系构造全局负样本
    neg_summary = "  ".join(f"{rel}={len(neg_edges_dict[rel])}" for rel in RELATIONS)
    print(f"[N_all] {neg_summary}")    #打印负样本数量

    if args.cv:
        cv_cfg = cfg["cv"]
        n_folds = int(cv_cfg["n_folds"])
        test_ratio = float(cv_cfg["test_ratio"])

        split_root = ensure_dir(proc_dir / f"cv{n_folds}_folds")
        _write_node2id(node2id_global, split_root)

        pos_test_edges, pos_fold_list = holdout_then_kfold(
            edges_dict,
            test_ratio=test_ratio,
            n_folds=n_folds,
            seed=seed,
        )  #对正样本做：holdout test + K-fold
        neg_test_edges, neg_fold_list = holdout_then_kfold(
            neg_edges_dict,
            test_ratio=test_ratio,
            n_folds=n_folds,
            seed=seed,
        )   #对全局负样本做同样的划分

        holdout_dir = ensure_dir(split_root / "holdout_test")
        for rel in RELATIONS:
            _save_edge_df(pos_test_edges[rel], holdout_dir / f"{rel}_test.txt")
            _save_edge_df(neg_test_edges[rel], holdout_dir / f"{rel}_test_neg_{NEG_TAG}.txt")

        for fold_k, (pos_scenario, neg_scenario) in enumerate(zip(pos_fold_list, neg_fold_list)):
            fold_dir = ensure_dir(split_root / f"fold_{fold_k}")

            _save_prepared_scenario(
                pos_scenario=pos_scenario,
                neg_scenario=neg_scenario,
                cache_dir=fold_dir,
                include_test=False,
                allow_train_val_overlap=False,
            )
            build_train_only_kg_from_split_dir(fold_dir)

        final_dir = ensure_dir(split_root / "final")
        final_pos_scenario = _make_final_scenario(pos_fold_list, pos_test_edges)
        final_neg_scenario = _make_final_scenario(neg_fold_list, neg_test_edges)

        _save_prepared_scenario(
            pos_scenario=final_pos_scenario,
            neg_scenario=final_neg_scenario,
            cache_dir=final_dir,
            include_test=True,
            allow_train_val_overlap=True,
        )
        build_train_only_kg_from_split_dir(final_dir)

if __name__ == "__main__":
    main()
