from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

#去掉节点前缀
def _strip_prefix(node_id: str) -> str:
    if node_id.startswith("Dr_"):
        return node_id[3:]
    if node_id.startswith("M_") or node_id.startswith("D_"):
        return node_id[2:]
    return node_id

#检查相似度矩阵是否合法：必须是二维矩阵、必须是方阵、必须对称、值域必须在0-1之间
def _check_similarity_matrix(S: np.ndarray, name: str) -> None:
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape={S.shape}")
    if not np.allclose(S, S.T, atol=1e-6):
        raise ValueError(f"{name} is not symmetric")
    if np.nanmin(S) < -1e-6 or np.nanmax(S) > 1.0 + 1e-6:
        raise ValueError(
            f"{name} values out of [0, 1]: min={float(np.nanmin(S))}, max={float(np.nanmax(S))}"
        )

#把缓存DS重新对齐到当前疾病
def _realign_cached_ds(
    DS_src: np.ndarray,
    src_ids: List[str],
    disease_raw_ids: Sequence[str],
) -> Tuple[np.ndarray, int, List[str]]:
    src_ids_norm = [sid.lower() for sid in src_ids]
    dst_ids_norm = [str(did).lower() for did in disease_raw_ids]
    src_idx = {sid: i for i, sid in enumerate(src_ids_norm)}

    D = len(dst_ids_norm)
    DS = np.eye(D, dtype=np.float32)
    matched = 0
    missing: List[str] = []

    for i, did in enumerate(dst_ids_norm):
        j = src_idx.get(did)
        if j is None:
            missing.append(str(disease_raw_ids[i]))
            continue
        matched += 1
        for k, did2 in enumerate(dst_ids_norm):
            j2 = src_idx.get(did2)
            if j2 is not None:
                DS[i, k] = float(DS_src[j, j2])

    return DS, matched, missing


def _load_cached_pymeshsim_wang_ds(
    disease_raw_ids: Sequence[str],
    cache_dir: Path,
    category: str,
    out_name: str,
) -> Optional[np.ndarray]:
    ds_npy_path = cache_dir / f"{out_name}.npy"
    ds_ids_path = cache_dir / f"{out_name}_ids.txt"
    ds_meta_path = cache_dir / f"{out_name}_meta.json"

    #如果三个文件不齐全就返回None
    if not (ds_npy_path.exists() and ds_ids_path.exists() and ds_meta_path.exists()):
        return None

    with open(ds_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    #检查meta文件内容是否符合：tool=pymeshsim，method=wang，category=c
    if str(meta.get("tool", "")).lower() != "pymeshsim":
        raise ValueError(f"Expected pyMeSHSim DS meta, got tool={meta.get('tool')}")
    if str(meta.get("method", "")).lower() != "wang":
        raise ValueError(f"Expected Wang DS meta, got method={meta.get('method')}")
    if str(meta.get("category", category)) != str(category):
        raise ValueError(
            f"Expected category={category}, got category={meta.get('category')} in {ds_meta_path}"
        )

    #读取DS矩阵，并检查矩阵是否合法
    DS_src = np.load(ds_npy_path).astype(np.float32)
    _check_similarity_matrix(DS_src, "Cached pyMeSHSim Wang DS")

    with open(ds_ids_path, "r", encoding="utf-8") as f:
        src_ids = [line.strip() for line in f if line.strip()]

    if DS_src.shape[0] != len(src_ids):
        raise ValueError(f"Cached DS shape {DS_src.shape} does not match ids length {len(src_ids)}")

    DS, matched, missing = _realign_cached_ds(DS_src, src_ids, disease_raw_ids)

    if missing:
        print(f"  missing ids  : {len(missing)} ")
    return DS


def compute_ds(disease_raw_ids: List[str]) -> np.ndarray:
    cache_dir = _PROJECT_ROOT / "data" / "external"
    out_name = "DS_pymeshsim_wang"
    cached = _load_cached_pymeshsim_wang_ds(
        disease_raw_ids=disease_raw_ids,
        cache_dir=cache_dir,
        category="C",
        out_name=out_name,
    )
    if cached is not None:
        return cached

    expected = [cache_dir / f"{out_name}.npy",cache_dir / f"{out_name}_ids.txt",cache_dir / f"{out_name}_meta.json",]
    missing = [str(p) for p in expected if not p.exists()]
    raise FileNotFoundError(
        f"Missing files: {missing}"
    )


def compute_gip(
    md_train: pd.DataFrame,
    microbe_raw_ids: List[str],
    disease_raw_ids: List[str],
    *,
    dm_train: Optional[pd.DataFrame] = None,
    dd_train: Optional[pd.DataFrame] = None,
    drug_raw_ids: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    M = len(microbe_raw_ids)
    D = len(disease_raw_ids)
    mid2idx = {m: i for i, m in enumerate(microbe_raw_ids)}
    did2idx = {d: i for i, d in enumerate(disease_raw_ids)}

    A_md = np.zeros((M, D), dtype=np.float32)
    for _, row in md_train.iterrows():
        mi = mid2idx.get(_strip_prefix(str(row["h"])))
        di = did2idx.get(_strip_prefix(str(row["t"])))
        if mi is not None and di is not None:
            A_md[mi, di] = 1.0

    GIP_M = _gip_kernel(A_md)
    #融合DM信息增强微生物GIP
    if dm_train is not None and drug_raw_ids:
        n_dr = len(drug_raw_ids)
        dr2idx = {d: i for i, d in enumerate(drug_raw_ids)}
        A_dm = np.zeros((n_dr, M), dtype=np.float32)
        for _, row in dm_train.iterrows():
            dri = dr2idx.get(_strip_prefix(str(row["h"])))
            mi = mid2idx.get(_strip_prefix(str(row["t"])))
            if dri is not None and mi is not None:
                A_dm[dri, mi] = 1.0
        GIP_M = ((GIP_M + _gip_kernel(A_dm.T)) / 2.0).astype(np.float32)

    GIP_D = _gip_kernel(A_md.T)
    #融合DD信息曾倩疾病GIP
    if dd_train is not None and drug_raw_ids:
        n_dr = len(drug_raw_ids)
        dr2idx = {d: i for i, d in enumerate(drug_raw_ids)}
        A_dd = np.zeros((n_dr, D), dtype=np.float32)
        for _, row in dd_train.iterrows():
            dri = dr2idx.get(_strip_prefix(str(row["h"])))
            di = did2idx.get(_strip_prefix(str(row["t"])))
            if dri is not None and di is not None:
                A_dd[dri, di] = 1.0
        GIP_D = ((GIP_D + _gip_kernel(A_dd.T)) / 2.0).astype(np.float32)

    return GIP_M, GIP_D


def compute_fs(
    md_train: pd.DataFrame,
    DS: np.ndarray,
    microbe_raw_ids: List[str],
    disease_raw_ids: List[str],
) -> np.ndarray:
    M = len(microbe_raw_ids)
    mid2idx = {m: i for i, m in enumerate(microbe_raw_ids)}
    did2idx = {d: i for i, d in enumerate(disease_raw_ids)}

    assoc: List[np.ndarray] = [np.array([], dtype=np.intp) for _ in range(M)]
    for _, row in md_train.iterrows():
        mi = mid2idx.get(_strip_prefix(str(row["h"])))
        di = did2idx.get(_strip_prefix(str(row["t"])))
        if mi is not None and di is not None:
            assoc[mi] = np.append(assoc[mi], di)
    assoc = [np.unique(a) for a in assoc]

    FS = np.eye(M, dtype=np.float32)
    for i in range(M):
        ai = assoc[i]
        if len(ai) == 0:
            continue
        for j in range(i + 1, M):
            aj = assoc[j]
            if len(aj) == 0:
                continue
            sub = DS[np.ix_(ai, aj)]
            s1 = float(sub.max(axis=1).sum())
            s2 = float(sub.max(axis=0).sum())
            fs = (s1 + s2) / (len(ai) + len(aj))
            FS[i, j] = FS[j, i] = fs
    return FS


def _gip_kernel(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    norms = np.sum(A ** 2, axis=1)
    gamma = n / max(np.sum(norms), 1e-10)
    gram = A @ A.T
    dist2 = norms[:, None] + norms[None, :] - 2 * gram
    K = np.exp(-gamma * np.maximum(dist2, 0)).astype(np.float32)
    zero_mask = (norms == 0)
    K[zero_mask, :] = 0.0
    K[:, zero_mask] = 0.0
    np.fill_diagonal(K, 1.0)
    return K


def compute_gip_drug(
    dm_train: pd.DataFrame,
    dd_train: pd.DataFrame,
    drug_raw_ids: List[str],
    microbe_raw_ids: List[str],
    disease_raw_ids: List[str],
) -> np.ndarray:
    n_dr = len(drug_raw_ids)
    dr2idx = {d: i for i, d in enumerate(drug_raw_ids)}
    m2idx = {m: i for i, m in enumerate(microbe_raw_ids)}
    d2idx = {d: i for i, d in enumerate(disease_raw_ids)}

    A_dm = np.zeros((n_dr, len(microbe_raw_ids)), dtype=np.float32)
    for _, row in dm_train.iterrows():
        dri = dr2idx.get(_strip_prefix(str(row["h"])))
        mi = m2idx.get(_strip_prefix(str(row["t"])))
        if dri is not None and mi is not None:
            A_dm[dri, mi] = 1.0

    A_dd = np.zeros((n_dr, len(disease_raw_ids)), dtype=np.float32)
    for _, row in dd_train.iterrows():
        dri = dr2idx.get(_strip_prefix(str(row["h"])))
        di = d2idx.get(_strip_prefix(str(row["t"])))
        if dri is not None and di is not None:
            A_dd[dri, di] = 1.0

    return ((_gip_kernel(A_dm) + _gip_kernel(A_dd)) / 2.0).astype(np.float32)

#计算药物 fingerprint 的 Tanimoto 相似度
def _tanimoto_matrix(X: np.ndarray) -> np.ndarray:
    dot = X @ X.T
    norms = np.sum(X ** 2, axis=1)
    denom = norms[:, None] + norms[None, :] - dot
    return np.where(denom > 0, dot / (denom + 1e-10), 0.0).astype(np.float32)


def compute_drug_similarity(
    drug_feat_dict: Dict[str, np.ndarray],
    all_drugs: List[str],
    morgan_dim: int,
) -> np.ndarray:
    n = len(all_drugs)
    raw_ids = [_strip_prefix(d) for d in all_drugs]
    X = np.zeros((n, max(morgan_dim, 1)), dtype=np.float32)
    for i, rid in enumerate(raw_ids):
        v = drug_feat_dict.get(rid)
        if v is not None:
            X[i, :min(len(v), morgan_dim)] = v[:morgan_dim]
    S_Dr = _tanimoto_matrix(X)
    np.fill_diagonal(S_Dr, 1.0)
    return S_Dr.astype(np.float32)

#原始相似度与GIP融合：如果原始相似度 s1 > 0， 用 s1，否则如果两个节点都有 GIP 信息：用 gip_beta * GIP,否则相似度设为0
def _fuse(s1: np.ndarray, s2: np.ndarray, beta: float = 0.3) -> np.ndarray:
    gip_offdiag = s2.copy()
    np.fill_diagonal(gip_offdiag, 0.0)
    has_degree = (gip_offdiag.sum(axis=1) > 0)
    both_have_degree = has_degree[:, None] & has_degree[None, :]
    result = np.where(s1 > 0, s1, np.where(both_have_degree, beta * s2, 0.0))
    np.fill_diagonal(result, 1.0)
    return result.astype(np.float32)

#自动计算KNN阈值：取非对角线、正相似度的第 percentile 分位数作为阈值，只保留相似度大于第 percentile的 KNN 边
def _compute_sim_threshold(S: np.ndarray, percentile: float = 25.0) -> float:
    N = S.shape[0]
    mask = ~np.eye(N, dtype=bool)
    vals = S[mask].flatten()
    vals = vals[vals > 0]
    if len(vals) == 0:
        return 0.0
    return float(np.percentile(vals, percentile))

#从相似度构建KNN边
def build_local_knn_edges(
    S: np.ndarray,
    knn_k: int,
    threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = S.shape[0]
    src, dst, wts = [], [], []
    real_k = min(knn_k, N - 1)
    if real_k <= 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0, dtype=np.float32)
    for i in range(N):
        row = S[i].copy()
        row[i] = -1.0
        topk_idx = np.argsort(row)[-real_k:][::-1]
        for j in topk_idx:
            if row[j] > threshold:
                src.append(j)
                dst.append(i)
                wts.append(float(row[j]))
    return (
        np.array(src, dtype=np.int64),
        np.array(dst, dtype=np.int64),
        np.array(wts, dtype=np.float32),#之前GAT加了边权重，现在没加
    )


def build_similarity_graphs(
    md_train: pd.DataFrame,
    all_microbes: List[str],
    all_diseases: List[str],
    all_drugs: List[str],
    drug_feat_dict: Dict[str, np.ndarray],
    knn_k: int,
    morgan_dim: int,
    *,
    dm_train: pd.DataFrame | None,
    dd_train: pd.DataFrame | None,
    use_drug_gip: bool,
    gip_beta: float = 0.3,
) -> Dict:
    #去掉节点前缀
    m_raw = [_strip_prefix(m) for m in all_microbes]
    d_raw = [_strip_prefix(d) for d in all_diseases]
    dr_raw = [_strip_prefix(d) for d in all_drugs]

    DS = compute_ds(d_raw) #读取疾病语义相似度
    GIP_M, GIP_D = compute_gip(
        md_train, m_raw, d_raw,
        dm_train=dm_train, dd_train=dd_train, drug_raw_ids=dr_raw,
    )  #计算GIP

    S_M = _fuse(compute_fs(md_train, DS, m_raw, d_raw), GIP_M, beta=gip_beta)
    S_D = _fuse(DS, GIP_D, beta=gip_beta)  #构建微生物相似度和疾病相似度

    S_Dr_chem = compute_drug_similarity(drug_feat_dict, all_drugs, morgan_dim)  #构建药物相似度
    if use_drug_gip and dm_train is not None and dd_train is not None:
        GIP_Dr = compute_gip_drug(dm_train, dd_train, dr_raw, m_raw, d_raw)
        S_Dr = _fuse(S_Dr_chem, GIP_Dr, beta=gip_beta)
    else:
        S_Dr = S_Dr_chem

    #构建KNN阈值和边
    tau_M = _compute_sim_threshold(S_M, percentile=35.0)
    tau_D = _compute_sim_threshold(S_D, percentile=35.0)
    tau_Dr = _compute_sim_threshold(S_Dr, percentile=35.0)
    print(f"KNN threshold：microbe={tau_M:.4f}  disease={tau_D:.4f}  drug={tau_Dr:.4f}")

    edges_mm = build_local_knn_edges(S_M, knn_k, threshold=tau_M)
    edges_dd = build_local_knn_edges(S_D, knn_k, threshold=tau_D)
    edges_drdr = build_local_knn_edges(S_Dr, knn_k, threshold=tau_Dr)
    print(f"KNN edges：microbe={len(edges_mm[0])}  disease={len(edges_dd[0])}  drug={len(edges_drdr[0])}")

    return {
        "S_M": S_M, "S_D": S_D, "S_Dr": S_Dr,
        "edges_mm": edges_mm, "edges_dd": edges_dd, "edges_drdr": edges_drdr,
    }


def _save_sim_data(data_fold_dir: str | Path, sim_data: Dict) -> None:
    data_fold_dir = Path(data_fold_dir)
    data_fold_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(data_fold_dir / "S_M.npy"), sim_data["S_M"])
    np.save(str(data_fold_dir / "S_D.npy"), sim_data["S_D"])
    np.save(str(data_fold_dir / "S_Dr.npy"), sim_data["S_Dr"])

    for key, fname in [("edges_mm", "edges_mm.npy"), ("edges_dd", "edges_dd.npy"), ("edges_drdr", "edges_drdr.npy")]:
        arr = sim_data.get(key)
        if arr is not None and len(arr) in (2, 3):
            np.save(str(data_fold_dir / fname), np.stack(arr, axis=0))
