from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import copy
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, matthews_corrcoef, roc_auc_score,
)

from src.config_utils import add_config_arg, load_config, ensure_dir, write_json
from src.metapath2vec import train_metapath2vec, load_embeddings_word2vec_txt
from src.external_node_features import load_external_features, load_drug_fingerprints
from src.similarity_features import build_similarity_graphs, _save_sim_data, _strip_prefix
from src.data_utils import (RELATIONS, NUM_KG_RELATIONS, ScenarioSplit,build_typed_edge_indices, load_scenario_from_dir, load_node2id, prepare_rel_data,build_node_type_tensor, NUM_NODE_TYPES)
from src.multi_task_decoder import MLPDecoder, UncertaintyWeighting, compute_multitask_loss
from src.dual_view_model import DualViewModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#固定随机种子
def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

#学习率调度函数，前warmup个epoch，学习率从很小逐渐升高，warmup后学习率按cosine曲线逐渐下降
def _cosine_warmup_lambda(epoch: int, warmup: int, total: int) -> float:
    if epoch < warmup:
        return max(epoch / max(warmup, 1), 0.01)
    progress = (epoch - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

#检查嵌入文件是否为空
def _emb_dim(emb: Dict[str, np.ndarray]) -> int:
    if not emb:
        raise ValueError("metapath2vec embeddings are empty.")
    return int(next(iter(emb.values())).shape[0])

#训练或者复用metapath2vec
def _train_embeddings(work_dir: Path, emb_cfg: Dict, seed: int) -> Path:
    emb_path = work_dir / "node_embeddings.txt"
    if emb_path.exists():
        return emb_path
    train_metapath2vec(
        kg_path=work_dir / "kg.edgelist",
        out_path=emb_path,
        dim=int(emb_cfg["dim"]),
        walk_length=int(emb_cfg["walk_length"]),
        num_walks=int(emb_cfg["num_walks"]),
        window=int(emb_cfg["window"]),
        epochs=int(emb_cfg["epochs"]),
        workers=int(emb_cfg["workers"]),
        negative=int(emb_cfg["negative"]),
        min_count=int(emb_cfg["min_count"]),
        seed=int(seed),
    )
    if not emb_path.exists():
        raise FileNotFoundError(f"Expected embeddings at {emb_path}")
    return emb_path

#评估函数
@torch.no_grad()
def _evaluate_z(
    decoder,
    z: torch.Tensor,
    pos_arr: np.ndarray,
    neg_arr: np.ndarray,
    rel: str,
    device: torch.device,
    threshold: float | None = None,
) -> Dict[str, float]:
    decoder.eval()
    pos_s = torch.from_numpy(pos_arr[0]).long().to(device)
    pos_d = torch.from_numpy(pos_arr[1]).long().to(device)
    neg_s = torch.from_numpy(neg_arr[0]).long().to(device)
    neg_d = torch.from_numpy(neg_arr[1]).long().to(device)

    pos_sc = torch.sigmoid(decoder(z, pos_s, pos_d, rel)).cpu().numpy()
    neg_sc = torch.sigmoid(decoder(z, neg_s, neg_d, rel)).cpu().numpy()

    scores = np.concatenate([pos_sc, neg_sc])
    labels = np.concatenate([np.ones_like(pos_sc), np.zeros_like(neg_sc)])

    nan_result = {
        "auroc": float("nan"), "aupr": float("nan"),
        "f1": float("nan"), "acc": float("nan"), "mcc": float("nan"),
        "sn": float("nan"), "sp": float("nan"), "best_threshold": 0.5,
    }
    if len(np.unique(labels)) < 2 or len(scores) == 0:
        return nan_result

    auroc = float(roc_auc_score(labels, scores))
    aupr = float(average_precision_score(labels, scores))

    #搜索最佳阈值
    if threshold is None:
        best_t, best_f1 = 0.5, -1.0
        for t in np.linspace(0.1, 0.9, 81):
            f = f1_score(labels, (scores >= t).astype(int), zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, float(t)
        threshold = best_t

    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "auroc": auroc,
        "aupr": aupr,
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "acc": float(accuracy_score(labels, preds)),
        "mcc": float(matthews_corrcoef(labels, preds)),
        "sn": float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan"),
        "sp": float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
        "best_threshold": threshold,
    }


def _train_v2(
    scenario_split: ScenarioSplit,
    node2id: Dict[str, int],
    emb: Dict[str, np.ndarray],
    emb_dim_mp2v: int,
    ext_taxonomy: Dict[str, np.ndarray] | None,
    ext_mesh: Dict[str, np.ndarray] | None,
    ext_drug: Dict[str, np.ndarray] | None,
    ext_dims: Dict,
    sim_data: Dict,
    kg_path: Path,
    cfg: Dict,
    seed: int,
    fixed_thresholds: Dict[str, float] | None = None,
    neg_cache_dir: "Path | None" = None,
) -> Dict[str, Dict[str, float]]:
    device = DEVICE
    _set_seed(seed)
    v2cfg = cfg["v2"]

    N = len(node2id)
    node_type = build_node_type_tensor(node2id, device)
    hidden_dim = int(v2cfg["hidden_dim"])
    emb_dim = int(v2cfg["emb_dim"])
    num_heads = int(v2cfg["num_heads"])
    struct_layers = int(v2cfg["struct_layers"])
    gat_layers = int(v2cfg["gat_layers"])
    dropout = float(v2cfg["dropout"])
    epochs = int(v2cfg["epochs"])
    lr = float(v2cfg["lr"])
    weight_decay = float(v2cfg["weight_decay"])
    patience = int(v2cfg["patience"])
    eval_every = int(v2cfg["eval_every"])
    warmup = int(v2cfg["warmup_epochs"])
    label_smooth = float(v2cfg["label_smooth"])
    loss_fusion = str(v2cfg.get("loss_fusion", "uw")).lower()
    aux_loss_weight = float(v2cfg.get("aux_loss_weight", 0.0))
    bio_kg_layers = int(v2cfg.get("bio_kg_layers", 0))

    m_nodes = sorted(n for n in node2id if n.startswith("M_"))
    d_nodes = sorted(n for n in node2id if n.startswith("D_"))
    dr_nodes = sorted(n for n in node2id if n.startswith("Dr_"))

    m_global = torch.tensor([node2id[n] for n in m_nodes], dtype=torch.long, device=device)
    d_global = torch.tensor([node2id[n] for n in d_nodes], dtype=torch.long, device=device)
    dr_global = torch.tensor([node2id[n] for n in dr_nodes], dtype=torch.long, device=device)

    x_mp2v = np.zeros((N, emb_dim_mp2v), dtype=np.float32)
    for node, idx in node2id.items():
        if node in emb:
            x_mp2v[idx] = emb[node]
    t_mp2v = torch.tensor(x_mp2v, dtype=torch.float32, device=device)

    tax_dim = ext_dims.get("taxonomy_dim", 0) if ext_taxonomy else 0
    mesh_dim = ext_dims.get("mesh_dim", 0) if ext_mesh else 0
    drug_dim = ext_dims.get("drug_dim", 0) if ext_drug else 0

    eff_tax_dim = max(tax_dim, 1)
    eff_mesh_dim = max(mesh_dim, 1)
    eff_drug_dim = max(drug_dim, 1)

    x_M = np.zeros((len(m_nodes), eff_tax_dim), dtype=np.float32)
    if ext_taxonomy and tax_dim > 0:
        for i, node in enumerate(m_nodes):
            v = ext_taxonomy.get(_strip_prefix(node))
            if v is not None and len(v) == tax_dim:
                x_M[i, :tax_dim] = v

    x_D = np.zeros((len(d_nodes), eff_mesh_dim), dtype=np.float32)
    if ext_mesh and mesh_dim > 0:
        for i, node in enumerate(d_nodes):
            v = ext_mesh.get(_strip_prefix(node))
            if v is not None and len(v) == mesh_dim:
                x_D[i, :mesh_dim] = v

    x_Dr = np.zeros((len(dr_nodes), eff_drug_dim), dtype=np.float32)
    if ext_drug and drug_dim > 0:
        for i, node in enumerate(dr_nodes):
            v = ext_drug.get(_strip_prefix(node))
            if v is not None and len(v) == drug_dim:
                x_Dr[i, :drug_dim] = v

    #转成GPU tensor
    t_M = torch.tensor(x_M, dtype=torch.float32, device=device)
    t_D = torch.tensor(x_D, dtype=torch.float32, device=device)
    t_Dr = torch.tensor(x_Dr, dtype=torch.float32, device=device)

    def _to_edge_index(key: str):
        e = sim_data[key]
        return (
            torch.tensor(e[0], dtype=torch.long, device=device),
            torch.tensor(e[1], dtype=torch.long, device=device),
        )

    sim_mm = _to_edge_index("edges_mm")
    sim_dd = _to_edge_index("edges_dd")
    sim_drdr = _to_edge_index("edges_drdr")
    kg_edges = build_typed_edge_indices(kg_path, node2id, device)

    model = DualViewModel(
        mp2v_dim=emb_dim_mp2v,
        tax_dim=eff_tax_dim,
        mesh_dim=eff_mesh_dim,
        drug_dim=eff_drug_dim,
        hidden_dim=hidden_dim,
        emb_dim=emb_dim,
        num_node_types=NUM_NODE_TYPES,
        num_kg_rels=NUM_KG_RELATIONS,
        num_heads=num_heads,
        struct_layers=struct_layers,
        gat_layers=gat_layers,
        dropout=dropout,
        bio_kg_layers=bio_kg_layers,
    ).to(device)

    out_dim = model.output_dim

    def _make_decoder():
        return MLPDecoder(out_dim).to(device)


    #创建辅助loss
    decoder = _make_decoder()
    use_aux = aux_loss_weight > 0.0
    decoder_struct = _make_decoder() if use_aux else None
    decoder_bio = _make_decoder() if use_aux else None

    model_params = (
        list(model.parameters())
        + list(decoder.parameters())
        + (list(decoder_struct.parameters()) if use_aux else [])
        + (list(decoder_bio.parameters()) if use_aux else [])
    )

    #这里增加了一个消融：训练的时候用用三个任务loss平均（mean）还是自适应融合（uw）
    if loss_fusion == "mean":
        uw = None
        opt = torch.optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)
        all_params = model_params
    else:
        uw = UncertaintyWeighting(list(RELATIONS)).to(device)
        uw_params = list(uw.parameters())

        uw_lr = float(v2cfg.get("uw_lr", 0.01))
        uw_weight_decay = float(v2cfg.get("uw_weight_decay", 0.0))

        opt = torch.optim.AdamW([
            {"params": model_params, "lr": lr, "weight_decay": weight_decay},
            {"params": uw_params, "lr": uw_lr, "weight_decay": uw_weight_decay},  #uw参数可以调整，在配置文件中
        ])
        all_params = model_params + uw_params

    #学习率调度器
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda ep: _cosine_warmup_lambda(ep, warmup, epochs),
    )

    rel_data = prepare_rel_data(scenario_split, node2id, cache_dir=neg_cache_dir)

    best_val, best_model_st, best_dec_st = -1.0, None, None
    best_val_per_rel: Dict[str, float] = {r: -1.0 for r in RELATIONS}
    no_improve = 0

    def _forward():
        return model(
            t_mp2v, t_M, t_D, t_Dr,
            kg_edges, sim_mm, sim_dd, sim_drdr,
            m_global, d_global, dr_global,
            node_type,
        )

    for ep in range(1, epochs + 1):
        model.train()
        decoder.train()
        if use_aux:
            decoder_struct.train()
            decoder_bio.train()

        z, z_struct, z_bio = _forward()
        loss = compute_multitask_loss(z, decoder, rel_data, device, label_smooth, uw)
        if use_aux:
            loss = loss + aux_loss_weight * compute_multitask_loss(
                z_struct, decoder_struct, rel_data, device, label_smooth, uw=None
            )
            loss = loss + aux_loss_weight * compute_multitask_loss(
                z_bio, decoder_bio, rel_data, device, label_smooth, uw=None
            )

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step()
        scheduler.step()

        if ep % eval_every == 0 or ep == epochs:
            model.eval()
            if use_aux:
                decoder_struct.eval()
                decoder_bio.eval()
            with torch.no_grad():
                zv, _, _ = _forward()

            val_aucs_per_rel: Dict[str, float] = {}
            for rel in RELATIONS:
                rd = rel_data[rel]
                if rd["va_pos"].shape[1] == 0:
                    continue
                vm = _evaluate_z(decoder, zv, rd["va_pos"], rd["va_neg"], rel, device)
                if not np.isnan(vm["auroc"]):
                    val_aucs_per_rel[rel] = vm["auroc"]

            if val_aucs_per_rel:
                avg_val = float(np.mean(list(val_aucs_per_rel.values())))
            else:
                avg_val = 0.0

            if avg_val > best_val:
                best_val = avg_val
                best_model_st = copy.deepcopy(model.state_dict())
                best_dec_st = copy.deepcopy(decoder.state_dict())
                for rel, auc in val_aucs_per_rel.items():
                    best_val_per_rel[rel] = auc
                no_improve = 0
            else:
                no_improve += eval_every

            if no_improve >= patience:
                break

    if best_model_st:
        model.load_state_dict(best_model_st)
        decoder.load_state_dict(best_dec_st)


    model.eval()
    with torch.no_grad():
        zf, _, _ = _forward()

    results: Dict[str, Dict[str, float]] = {}
    for rel in RELATIONS:
        rd = rel_data[rel]
        threshold = fixed_thresholds.get(rel) if fixed_thresholds else None
        use_te = rd["te_pos"].shape[1] > 0
        pos_arr = rd["te_pos"] if use_te else rd["va_pos"]
        neg_arr = rd["te_neg"] if use_te else rd["va_neg"]
        m = _evaluate_z(decoder, zf, pos_arr, neg_arr, rel, device, threshold=threshold)
        m["best_val_auc"] = round(best_val, 6)
        m["best_val_auc_rel"] = round(best_val_per_rel.get(rel, float("nan")), 6)
        m["n_test_pos"] = int(pos_arr.shape[1])
        results[rel] = m  #五折结果是每折的验证集结果，final是独立测试集结果，独立测试集使用五折的平均阈值

    return results

#五折+holdout
def _run_v2_holdout_cv(
    base: Path,
    cfg: Dict,
    ext_taxonomy,
    ext_mesh,
    ext_drug,
    ext_dims: Dict,
    drug_fp_dict: Dict,
    out_root: Path,
    n_folds: int,
    *,
    use_drug_gip: bool,
    gip_beta: float = 0.3,
):
    seed = cfg["seed"]
    ext_cfg = cfg["external_features"]
    v2_sim_cfg = cfg["v2"]["similarity"]
    v2_knn_k = int(v2_sim_cfg["knn_k"])
    morgan_dim = int(ext_cfg["morgan_nbits"])

    split_root = base / cfg["paths"]["processed_dir"] / f"cv{n_folds}_folds"
    final_dir = split_root / "final"
    node2id_global = load_node2id(split_root)
    combo = "v2__multi_task__mlp"
    all_fold_results: Dict[str, List[Dict[str, Dict[str, float]]]] = {}
    all_microbes = sorted(n for n in node2id_global if n.startswith("M_"))
    all_diseases = sorted(n for n in node2id_global if n.startswith("D_"))
    all_drugs = sorted(n for n in node2id_global if n.startswith("Dr_"))

    for fold_k in range(n_folds):
        fold_dir = split_root / f"fold_{fold_k}"
        scenario_split = load_scenario_from_dir(fold_dir, include_test=False)  #五折阶段不读取test
        kg_path = fold_dir / "kg.edgelist"

        emb_path = _train_embeddings(fold_dir, cfg["embeddings"], seed=seed + fold_k * 100)
        emb = load_embeddings_word2vec_txt(str(emb_path))
        emb_dim_mp2v = _emb_dim(emb)

        sim_data = build_similarity_graphs(
            md_train=scenario_split.splits["MD"].train,
            all_microbes=all_microbes,
            all_diseases=all_diseases,
            all_drugs=all_drugs,
            drug_feat_dict=drug_fp_dict,
            knn_k=v2_knn_k,
            morgan_dim=morgan_dim,
            dm_train=scenario_split.splits["DM"].train,
            dd_train=scenario_split.splits["DD"].train,
            use_drug_gip=use_drug_gip,
            gip_beta=gip_beta,
        )
        _save_sim_data(fold_dir, sim_data)

        results = _train_v2(
            scenario_split=scenario_split,
            node2id=node2id_global,
            emb=emb,
            emb_dim_mp2v=emb_dim_mp2v,
            ext_taxonomy=ext_taxonomy,
            ext_mesh=ext_mesh,
            ext_drug=ext_drug,
            ext_dims=ext_dims,
            sim_data=sim_data,
            kg_path=kg_path,
            cfg=cfg,
            seed=seed + fold_k * 100,
            neg_cache_dir=fold_dir,
        )
        all_fold_results.setdefault(combo, []).append(results)
        _print_fold_results(fold_k, n_folds, results)

    #计算五折均值
    fold_thresholds: Dict[str, Dict[str, float]] = {}
    for combo, fold_res_list in all_fold_results.items():
        fold_thresholds[combo] = {}
        for rel in RELATIONS:
            ts = [
                fr[rel].get("best_threshold", 0.5)
                for fr in fold_res_list
                if rel in fr and not np.isnan(fr[rel].get("best_threshold", float("nan")))
            ]
            fold_thresholds[combo][rel] = float(np.mean(ts)) if ts else 0.5

    cv_summary = _write_cv_summary(all_fold_results, out_root)

    final_scenario = load_scenario_from_dir(final_dir, include_test=True)
    kg_path = final_dir / "kg.edgelist"

    emb_path = _train_embeddings(final_dir, cfg["embeddings"], seed=seed)
    emb = load_embeddings_word2vec_txt(str(emb_path))
    emb_dim_mp2v = _emb_dim(emb)

    sim_data = build_similarity_graphs(
        md_train=final_scenario.splits["MD"].train,
        dm_train=final_scenario.splits["DM"].train,
        dd_train=final_scenario.splits["DD"].train,
        all_microbes=all_microbes,
        all_diseases=all_diseases,
        all_drugs=all_drugs,
        drug_feat_dict=drug_fp_dict,
        knn_k=v2_knn_k,
        morgan_dim=morgan_dim,
        use_drug_gip=use_drug_gip,
        gip_beta=gip_beta,
    )

    final_results: Dict = {
        combo: _train_v2(
            scenario_split=final_scenario,
            node2id=node2id_global,
            emb=emb,
            emb_dim_mp2v=emb_dim_mp2v,
            ext_taxonomy=ext_taxonomy,
            ext_mesh=ext_mesh,
            ext_drug=ext_drug,
            ext_dims=ext_dims,
            sim_data=sim_data,
            kg_path=kg_path,
            cfg=cfg,
            seed=seed,
            fixed_thresholds=fold_thresholds.get(combo),
            neg_cache_dir=final_dir,
        )
    }
    _print_holdout_summary(final_results)
    return cv_summary, final_results


def _print_fold_results(fold_k, n_folds, results):
    print(f"\nfold{fold_k + 1}")
    for rel in RELATIONS:
        print(f"  {rel}  {_fmt_metrics(results.get(rel, {}))}")


_CV_METRICS = ["auroc", "aupr", "f1", "acc", "mcc", "sn", "sp"]


def _fmt_metrics(m: Dict) -> str:
    return (
        f"AUROC={m.get('auroc', float('nan')):.4f}  "
        f"AUPR={m.get('aupr', float('nan')):.4f}  "
        f"F1={m.get('f1', float('nan')):.4f}  "
        f"ACC={m.get('acc', float('nan')):.4f}  "
        f"MCC={m.get('mcc', float('nan')):.4f}  "
        f"SN={m.get('sn', float('nan')):.4f}  "
        f"SP={m.get('sp', float('nan')):.4f}"
    )


def _write_cv_summary(all_fold_results, out_dir):
    summary: Dict = {}
    rows = []

    for combo_name, fold_list in all_fold_results.items():
        combo_summary: Dict[str, Dict] = {}
        for rel in RELATIONS:
            metric_vals = {m: [] for m in _CV_METRICS}
            for fold_res in fold_list:
                if rel not in fold_res:
                    continue
                for metric in _CV_METRICS:
                    value = fold_res[rel].get(metric, float("nan"))
                    if not np.isnan(value):
                        metric_vals[metric].append(value)

            agg: Dict[str, float] = {}
            for metric, values in metric_vals.items():
                agg[f"{metric}_mean"] = float(np.mean(values)) if values else float("nan")
                agg[f"{metric}_std"] = float(np.std(values)) if values else float("nan")
            agg["n_folds_valid"] = len(metric_vals["auroc"])
            combo_summary[rel] = agg
            rows.append({"relation": rel, **agg})

        summary[combo_name] = combo_summary

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "cv_comparison_table.csv", index=False)

    print("\nCV mean±std")
    for _, row in df.iterrows():
        ms = {k: row.get(f"{k}_mean", float("nan")) for k in _CV_METRICS}
        ss = {k: row.get(f"{k}_std", float("nan")) for k in _CV_METRICS}
        parts = "  ".join(f"{k.upper()}={ms[k]:.4f}±{ss[k]:.4f}" for k in _CV_METRICS)
        print(f"  {row['relation']}  {parts}")
    return summary


def _print_holdout_summary(final_results):
    print("\n[Hold-out Test]")
    for res in final_results.values():
        for rel in RELATIONS:
            m = res.get(rel, {})
            print(f"  {rel}  {_fmt_metrics(m)}  n={m.get('n_test_pos', 0)}")


def main():
    ap = argparse.ArgumentParser()
    add_config_arg(ap)
    args = ap.parse_args()

    base = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config)
    seed = cfg["seed"]

    out_root = ensure_dir(base / cfg["paths"]["outputs_dir"])

    ext_cfg = cfg.get("external_features", {})
    ext_taxonomy, ext_mesh, ext_drug, ext_dims = None, None, None, {}

    _ext_dir = base / ext_cfg["cache_dir"]
    drug_fp_dict = load_drug_fingerprints(_ext_dir)

    #如果开启外部特征就加载微生物taxonomy、疾病Mesh
    if bool(ext_cfg.get("enabled", True)):
        ext_taxonomy, ext_mesh, _, ext_dims = load_external_features(_ext_dir)
        if drug_fp_dict:
            fp_dim = int(len(next(iter(drug_fp_dict.values()))))
            ext_drug = drug_fp_dict
            ext_dims["drug_dim"] = fp_dim
            print(f"Morgan fingerprint, dim={fp_dim}.")
        else:
            print("drug_fingerprints.pkl not found.")
    else:
        print("enabled=False.")

    v2_sim_cfg = cfg["v2"]["similarity"]
    use_drug_gip = bool(v2_sim_cfg["use_drug_gip"])
    gip_beta = float(v2_sim_cfg.get("gip_beta", 0.3))

    _v2 = cfg["v2"]
    print(
        f"[Params] lr={_v2['lr']}  aux_loss_weight={_v2.get('aux_loss_weight', 0.0)}  "
        f"loss_fusion={_v2.get('loss_fusion', 'uw')}  "
        f"uw_lr={_v2.get('uw_lr', 0.01)}  "
        f"hidden_dim={_v2['hidden_dim']}  emb_dim={_v2['emb_dim']}  "
        f"num_heads={_v2['num_heads']}  dropout={_v2['dropout']}  "
        f"label_smooth={_v2['label_smooth']}  knn_k={_v2['similarity']['knn_k']}  "
        f"gip_beta={gip_beta}"
    )

    cv_cfg = cfg["cv"]
    n_folds = int(cv_cfg["n_folds"])
    cv_out = ensure_dir(out_root / f"cv_{n_folds}fold")

    cv_summary, final_results = _run_v2_holdout_cv(
        base=base,
        cfg=cfg,
        ext_taxonomy=ext_taxonomy,
        ext_mesh=ext_mesh,
        ext_drug=ext_drug,
        ext_dims=ext_dims,
        drug_fp_dict=drug_fp_dict,
        out_root=cv_out,
        n_folds=n_folds,
        use_drug_gip=use_drug_gip,
        gip_beta=gip_beta,
    )

    write_json(cv_summary, cv_out / "cv_results.json")
    write_json(final_results, cv_out / "final_holdout_results.json")
    print(f"\n[Done] results saved to {cv_out}")


if __name__ == "__main__":
    main()
