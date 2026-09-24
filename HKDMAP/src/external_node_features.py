from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import tarfile
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import re
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


TAXONOMY_RANKS = ["phylum", "class", "order", "family", "genus", "species"]
MESH_CATS = list("ABCDEFGHIJKLMNVZ")

PUBCHEM_SMILES_KEYS = [
    "CanonicalSMILES",
    "IsomericSMILES",
    "ConnectivitySMILES",
]

#找出合法数字 TaxID，且出现在NCBI taxdump 的 nodes.dmp 里
def _resolve_taxid_strict(raw: str, taxid2parent: Dict[int, int]) -> Optional[int]:
    s = str(raw).strip()
    if not s.isdigit():
        return None
    tid = int(s)
    return tid if tid in taxid2parent else None

#找出合法的MeSH Descriptor/Supplemental ID
def _resolve_mesh_id_strict(raw: str) -> Optional[str]:
    s = str(raw).strip().upper()
    return s if re.fullmatch(r"[DC]\d{6}(\d{3})?", s) else None

#找出合法PubChem CID
def _resolve_pubchem_cid_strict(raw: str) -> Optional[str]:
    s = str(raw).strip()
    return s if s.isdigit() else None

#如果 names.dmp 和 nodes.dmp 已存在，直接复用，否则下载 taxdump.tar.gz，只解压 names.dmp 和 nodes.dmp，解压后删除 tar.gz
def _download_taxdump(cache_dir: Path) -> Path:
    extract_dir = cache_dir / "taxdump"
    names_file = extract_dir / "names.dmp"
    nodes_file = extract_dir / "nodes.dmp"

    if names_file.exists() and nodes_file.exists():
        return extract_dir

    url = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"
    tar_path = cache_dir / "taxdump.tar.gz"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [Taxonomy] Downloading {url} ...")
    urllib.request.urlretrieve(url, str(tar_path))

    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(tar_path), "r:gz") as tf:
        for member in tf.getmembers():
            if member.name in ("names.dmp", "nodes.dmp"):
                member.name = os.path.basename(member.name)
                tf.extract(member, str(extract_dir))

    if tar_path.exists():
        tar_path.unlink()

    return extract_dir

#解析names.dmp → taxid2name，nodes.dmp → taxid2parent / taxid2rank
def _parse_taxdump(
    taxdump_dir: Path,
) -> Tuple[Dict[int, str], Dict[int, int], Dict[int, str]]:
    taxid2name: Dict[int, str] = {}
    with open(taxdump_dir / "names.dmp", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4 and parts[3] == "scientific name":
                taxid2name[int(parts[0])] = parts[1]

    taxid2parent: Dict[int, int] = {}
    taxid2rank: Dict[int, str] = {}
    with open(taxdump_dir / "nodes.dmp", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.split("|")]
            tid = int(parts[0])
            taxid2parent[tid] = int(parts[1])
            taxid2rank[tid] = parts[2]

    return taxid2name, taxid2parent, taxid2rank

#沿着父节点一直想上找lineage，并且只保留TAXONOMY_RANKS = ["phylum", "class", "order", "family", "genus", "species"]
def _get_lineage(
    taxid: int,
    taxid2parent: Dict[int, int],
    taxid2rank: Dict[int, str],
    taxid2name: Dict[int, str],
) -> Dict[str, str]:
    lineage: Dict[str, str] = {}
    current = taxid
    visited: Set[int] = set()

    while current != 1 and current in taxid2parent and current not in visited:
        visited.add(current)
        rank = taxid2rank.get(current, "")
        if rank in TAXONOMY_RANKS:
            lineage[rank] = taxid2name.get(current, f"unknown_{current}")
        current = taxid2parent[current]

    return lineage

#下载并解析txsdump，对对 microbe_ids 解析 TaxID，找 lineage，每个 rank 统计 top_k vocabulary，每个 rank 多一个 other 位，每个 microbe 生成 one-hot taxonomy 特征，保存 taxonomy_meta.json
def build_taxonomy_features_onehot(
    microbe_ids: List[str],
    cache_dir: Path,
    top_k: Dict[str, int],
) -> Tuple[Dict[str, np.ndarray], int]:
    taxdump_dir = _download_taxdump(cache_dir)
    taxid2name, taxid2parent, taxid2rank = _parse_taxdump(taxdump_dir)

    lineages: Dict[str, Dict[str, str]] = {}
    for mid in microbe_ids:
        tid = _resolve_taxid_strict(mid, taxid2parent)
        if tid is not None:
            lineages[mid] = _get_lineage(tid, taxid2parent, taxid2rank, taxid2name)

    print(f"valid TaxID lineage: {len(lineages)}/{len(microbe_ids)}")

    rank_counters: Dict[str, Counter] = {r: Counter() for r in TAXONOMY_RANKS}
    for lin in lineages.values():
        for rank, name in lin.items():
            rank_counters[rank][name] += 1

    rank_vocab: Dict[str, Dict[str, int]] = {}
    for rank in TAXONOMY_RANKS:
        k = int(top_k[rank])
        most_common = rank_counters[rank].most_common(k)
        rank_vocab[rank] = {name: i for i, (name, _) in enumerate(most_common)}

    feat_dim = sum(len(rank_vocab[r]) + 1 for r in TAXONOMY_RANKS)

    features: Dict[str, np.ndarray] = {}
    for mid in microbe_ids:
        feat = np.zeros(feat_dim, dtype=np.float32)
        if mid in lineages:
            offset = 0
            for rank in TAXONOMY_RANKS:
                vocab = rank_vocab[rank]
                taxon = lineages[mid].get(rank)
                if taxon is not None and taxon in vocab:
                    feat[offset + vocab[taxon]] = 1.0
                elif taxon is not None:
                    feat[offset + len(vocab)] = 1.0
                offset += len(vocab) + 1
        features[mid] = feat

    meta = {"rank_vocab": {r: list(v.keys()) for r, v in rank_vocab.items()}}
    with open(cache_dir / "taxonomy_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Taxonomy: feature_dim={feat_dim}")
    return features, feat_dim

#提取mesh id
def _extract_mesh_ids(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        mid = value.split("/")[-1].strip()
        return [mid] if mid else []
    if isinstance(value, dict):
        for key in ("@id", "id", "identifier"):
            if key in value:
                mid = str(value[key]).split("/")[-1].strip()
                return [mid] if mid else []
        return []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            out.extend(_extract_mesh_ids(item))
        return out
    return []

#调用 MeSH JSON API 获取 treeNumber
def _fetch_mesh_tree_numbers(
    mesh_id: str,
    cache: Dict[str, List[str]],
    delay: float = 0.35,
    retries: int = 3,
    timeout: int = 20,
) -> List[str]:
    cached = cache.get(mesh_id)
    if cached:
        return cached
    url = f"https://id.nlm.nih.gov/mesh/{mesh_id}.json"
    for attempt in range(1, retries + 1):
        tree_numbers: List[str] = []
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            for tn in _extract_mesh_ids(data.get("treeNumber", [])):
                if tn:
                    tree_numbers.append(tn)

            if not tree_numbers:
                for field in ("preferredMappedTo", "mappedTo"):
                    for mapped_id in _extract_mesh_ids(data.get(field, [])):
                        if mapped_id.startswith("D"):
                            tree_numbers.extend(
                                _fetch_mesh_tree_numbers(mapped_id, cache, delay=delay, retries=retries, timeout=timeout)
                            )
                    if tree_numbers:
                        break

            if tree_numbers:
                tree_numbers = sorted(set(tree_numbers))
                cache[mesh_id] = tree_numbers
                time.sleep(delay)
                return tree_numbers

            time.sleep(delay)
            return []

        except Exception as exc:
            if attempt == retries:
                print(f"{mesh_id} fetch failed after {retries} attempts: {exc}")
                time.sleep(delay)
                return []
            time.sleep(delay * attempt)

    return []

#本地MeSH XML 解析部分
def _xml_tag_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _find_child_text(elem: ET.Element, child_name: str) -> Optional[str]:
    for child in elem:
        if _xml_tag_name(child.tag) == child_name and child.text:
            return child.text.strip()
    return None


def _find_first_xml(mesh_xml_dir: Path, prefix: str) -> Optional[Path]:
    matches = sorted(mesh_xml_dir.glob(f"{prefix}*.xml"))
    return matches[0] if matches else None


def _parse_mesh_descriptor_xml(desc_xml: Path) -> Dict[str, List[str]]:
    mesh_tree: Dict[str, List[str]] = {}
    for _event, elem in ET.iterparse(desc_xml, events=("end",)):
        if _xml_tag_name(elem.tag) != "DescriptorRecord":
            continue
        ui = _find_child_text(elem, "DescriptorUI")
        if not ui:
            elem.clear()
            continue
        tree_numbers: List[str] = []
        for node in elem.iter():
            if _xml_tag_name(node.tag) == "TreeNumber" and node.text:
                tree_numbers.append(node.text.strip())
        if tree_numbers:
            mesh_tree[ui] = sorted(set(tree_numbers))
        elem.clear()
    return mesh_tree


def _parse_mesh_supplement_xml(supp_xml: Path, mesh_tree: Dict[str, List[str]]) -> int:
    mapped_count = 0
    for _event, elem in ET.iterparse(supp_xml, events=("end",)):
        if _xml_tag_name(elem.tag) != "SupplementalRecord":
            continue
        cui = _find_child_text(elem, "SupplementalRecordUI")
        if not cui:
            elem.clear()
            continue
        desc_ids: List[str] = []
        for node in elem.iter():
            if _xml_tag_name(node.tag) == "DescriptorUI" and node.text:
                did = node.text.strip()
                if did.startswith("D"):
                    desc_ids.append(did)
        tree_numbers: List[str] = []
        for did in sorted(set(desc_ids)):
            tree_numbers.extend(mesh_tree.get(did, []))
        if tree_numbers:
            mesh_tree[cui] = sorted(set(tree_numbers))
            mapped_count += 1
        elem.clear()
    return mapped_count


def _load_mesh_tree_from_local_xml(cache_dir: Path) -> Optional[Dict[str, List[str]]]:
    mesh_xml_dir = cache_dir / "mesh_xml"
    desc_xml = _find_first_xml(mesh_xml_dir, "desc")
    supp_xml = _find_first_xml(mesh_xml_dir, "supp")

    if desc_xml is None:
        return None

    xml_cache_path = cache_dir / "mesh_tree_from_xml_cache.json"
    if xml_cache_path.exists():
        with open(xml_cache_path, encoding="utf-8") as f:
            return json.load(f)

    print(f"Parsing {desc_xml}")
    mesh_tree = _parse_mesh_descriptor_xml(desc_xml)

    if supp_xml is not None:
        print(f"Parsing {supp_xml}")
        mapped_count = _parse_mesh_supplement_xml(supp_xml, mesh_tree)
        print(f"SCR records mapped to descriptor trees: {mapped_count}")
    else:
        print("supp*.xml not found; C-type IDs may remain zero")

    with open(xml_cache_path, "w", encoding="utf-8") as f:
        json.dump(mesh_tree, f, indent=2)

    print(f"IDs with tree numbers: {len(mesh_tree)}")
    return mesh_tree

#过滤合法Mesh id，优先从本地获取tree number，没有XML时走Mesh api，统计 level0/level1/level2/level3 vocabulary，生成 multi-hot 特征
def build_mesh_features(
    disease_ids: List[str],
    cache_dir: Path,
    level1_top_k: int,
    level2_top_k: int = 120,
    level3_top_k: int = 200,
) -> Tuple[Dict[str, np.ndarray], int]:
    id_map: Dict[str, str] = {}
    for did in disease_ids:
        ui = _resolve_mesh_id_strict(did)
        if ui is not None:
            id_map[did] = ui
    mesh_ids = sorted(set(id_map.values()))
    print(f"valid MeSH IDs: {len(mesh_ids)}/{len(disease_ids)}")
    xml_tree = _load_mesh_tree_from_local_xml(cache_dir)

    if xml_tree is not None:
        all_tree = {mid: xml_tree.get(mid, []) for mid in mesh_ids}
    else:
        cache_path = cache_dir / "mesh_cache.json"
        cache: Dict[str, List[str]] = {}
        if cache_path.exists():
            with open(cache_path, encoding="utf-8") as f:
                cache = json.load(f)
            cache = {str(k): v for k, v in cache.items() if v}

        all_tree: Dict[str, List[str]] = {}
        for i, mid in enumerate(mesh_ids):
            all_tree[mid] = _fetch_mesh_tree_numbers(mid, cache)
            if (i + 1) % 100 == 0:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f)

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f)

    level1_counter: Counter = Counter()
    level2_counter: Counter = Counter()
    level3_counter: Counter = Counter()

    for tns in all_tree.values():
        for tn in tns:
            parts = tn.split(".")
            if not parts:
                continue
            level1_counter[parts[0]] += 1
            if len(parts) >= 2:
                level2_counter[".".join(parts[:2])] += 1
            if len(parts) >= 3:
                level3_counter[".".join(parts[:3])] += 1

    level0_vocab = {c: i for i, c in enumerate(MESH_CATS)}
    level1_vocab = {code: i for i, (code, _) in enumerate(level1_counter.most_common(int(level1_top_k)))}
    level2_vocab = {code: i for i, (code, _) in enumerate(level2_counter.most_common(int(level2_top_k)))}
    level3_vocab = {code: i for i, (code, _) in enumerate(level3_counter.most_common(int(level3_top_k)))}

    level0_dim = len(level0_vocab)
    level1_dim = len(level1_vocab) + 1
    level2_dim = len(level2_vocab) + 1
    level3_dim = len(level3_vocab) + 1

    offset1 = level0_dim
    offset2 = offset1 + level1_dim
    offset3 = offset2 + level2_dim
    feat_dim = level0_dim + level1_dim + level2_dim + level3_dim

    print(f"vocab: level1={len(level1_vocab)}, level2={len(level2_vocab)}, level3={len(level3_vocab)}, total_dim={feat_dim}")

    features: Dict[str, np.ndarray] = {}
    for did in disease_ids:
        feat = np.zeros(feat_dim, dtype=np.float32)
        ui = id_map.get(did)
        if ui and ui in all_tree:
            for tn in all_tree[ui]:
                parts = tn.split(".")
                if not parts:
                    continue
                cat_letter = parts[0][0] if parts[0] else ""
                if cat_letter in level0_vocab:
                    feat[level0_vocab[cat_letter]] = 1.0
                level1_code = parts[0]
                if level1_code in level1_vocab:
                    feat[offset1 + level1_vocab[level1_code]] = 1.0
                else:
                    feat[offset1 + len(level1_vocab)] = 1.0
                if len(parts) >= 2:
                    level2_code = ".".join(parts[:2])
                    if level2_code in level2_vocab:
                        feat[offset2 + level2_vocab[level2_code]] = 1.0
                    else:
                        feat[offset2 + len(level2_vocab)] = 1.0
                if len(parts) >= 3:
                    level3_code = ".".join(parts[:3])
                    if level3_code in level3_vocab:
                        feat[offset3 + level3_vocab[level3_code]] = 1.0
                    else:
                        feat[offset3 + len(level3_vocab)] = 1.0
        features[did] = feat

    n_nonzero = sum(1 for f in features.values() if f.any())
    mesh_meta = {
        "encoding": "mesh_level0_level1_level2_level3_multihot",
        "level0_vocab": list(level0_vocab.keys()),
        "level1_vocab": list(level1_vocab.keys()),
        "level2_vocab": list(level2_vocab.keys()),
        "level3_vocab": list(level3_vocab.keys()),
        "level1_top_k": int(level1_top_k),
        "level2_top_k": int(level2_top_k),
        "level3_top_k": int(level3_top_k),
    }
    with open(cache_dir / "mesh_feature_meta.json", "w", encoding="utf-8") as f:
        json.dump(mesh_meta, f, indent=2)

    print(f"Mesh：feature_dim={feat_dim}, nonzero={n_nonzero}/{len(disease_ids)}")
    return features, feat_dim


def _fetch_pubchem_props_batch(
    cids: List[str],
    cache: Dict[str, Dict],
    batch_size: int = 100,
) -> Dict[str, Dict]:
    fields = ",".join(PUBCHEM_SMILES_KEYS)

    to_fetch = [c for c in cids if c not in cache]
    if not to_fetch:
        return cache

    print(f"fetching properties for {len(to_fetch)} CIDs")

    for i in range(0, len(to_fetch), batch_size):
        batch = to_fetch[i: i + batch_size]
        cid_str = ",".join(batch)
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
            f"{cid_str}/property/{fields}/JSON"
        )
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            for prop in data.get("PropertyTable", {}).get("Properties", []):
                cid = str(prop.get("CID", ""))
                if cid:
                    cache[cid] = prop
        except Exception as exc:
            print(f"    [PubChem] batch {i}-{i + len(batch)} error: {exc}")
        time.sleep(0.4)

    return cache


def build_drug_features(
    drug_ids: List[str],
    cache_dir: Path,
    morgan_nbits: int,
    morgan_radius: int,
) -> Tuple[Dict[str, np.ndarray], int]:
    try:
        from rdkit import Chem
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
        has_rdkit = True
    except ImportError:
        has_rdkit = False

    props_cache_path = cache_dir / "pubchem_props_cache.json"
    props_cache: Dict[str, Dict] = {}
    if props_cache_path.exists():
        with open(props_cache_path, encoding="utf-8") as f:
            props_cache = json.load(f)

    drug_to_cid = {d: _resolve_pubchem_cid_strict(d) for d in drug_ids}
    numeric_cids = sorted({c for c in drug_to_cid.values() if c is not None})

    print(
        f"valid numeric PubChem CIDs: "
        f"{sum(1 for d in drug_ids if drug_to_cid.get(d))}/{len(drug_ids)} "
        f"({len(numeric_cids)} unique)"
    )

    props_cache = _fetch_pubchem_props_batch(numeric_cids, props_cache)

    with open(props_cache_path, "w", encoding="utf-8") as f:
        json.dump(props_cache, f)

    fp_dim = int(morgan_nbits) if has_rdkit else 0
    morgan_gen = GetMorganGenerator(radius=int(morgan_radius), fpSize=fp_dim) if has_rdkit else None

    raw_fp: Dict[str, np.ndarray] = {}

    for did in drug_ids:
        cid = drug_to_cid.get(did)
        if cid is None:
            continue
        prop = props_cache.get(cid)
        if prop is None:
            continue
        fp_arr = np.zeros(fp_dim, dtype=np.float32)
        if has_rdkit and morgan_gen is not None:
            smiles = (
                prop.get("CanonicalSMILES")
                or prop.get("IsomericSMILES")
                or prop.get("ConnectivitySMILES")
                or ""
            )
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp_arr = np.array(morgan_gen.GetFingerprint(mol), dtype=np.float32)
        raw_fp[did] = fp_arr

    fp_dict = {
        did: raw_fp.get(did, np.zeros(fp_dim, dtype=np.float32))
        for did in drug_ids
    }

    print(f"fp_dim={fp_dim}, nonzero={len(raw_fp)}/{len(drug_ids)}")
    return fp_dict, fp_dim


def collect_unique_ids(
    processed_dir: Path,
) -> Tuple[List[str], List[str], List[str]]:
    microbe_ids: Set[str] = set()
    disease_ids: Set[str] = set()
    drug_ids: Set[str] = set()

    md_path = processed_dir / "microbe_disease.txt"
    if md_path.exists():
        df = pd.read_csv(md_path, sep="\t", header=None, dtype=str)
        microbe_ids.update(df[0].tolist())
        disease_ids.update(df[1].tolist())

    dm_path = processed_dir / "drug_microbe.txt"
    if dm_path.exists():
        df = pd.read_csv(dm_path, sep="\t", header=None, dtype=str)
        drug_ids.update(df[0].tolist())
        microbe_ids.update(df[1].tolist())

    dd_path = processed_dir / "drug_disease_filt.txt"
    if dd_path.exists():
        df = pd.read_csv(dd_path, sep="\t", header=None, dtype=str)
        drug_ids.update(df[0].tolist())
        disease_ids.update(df[1].tolist())

    return sorted(microbe_ids), sorted(disease_ids), sorted(drug_ids)


def load_external_features(
    ext_dir: str | Path,
) -> Tuple[
    Optional[Dict[str, np.ndarray]],
    Optional[Dict[str, np.ndarray]],
    Optional[Dict[str, np.ndarray]],
    Dict,
]:
    ext_dir = Path(ext_dir)
    dims_path = ext_dir / "feature_dims.json"

    if not dims_path.exists():
        return None, None, None, {}

    with open(dims_path, encoding="utf-8") as f:
        dims = json.load(f)

    def _load_pkl(name: str) -> Optional[Dict[str, np.ndarray]]:
        path = ext_dir / name
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    tax = _load_pkl("taxonomy_features.pkl")
    mesh = _load_pkl("mesh_features.pkl")
    drug = _load_pkl("drug_fingerprints.pkl")

    return tax, mesh, drug, dims


def load_drug_fingerprints(ext_dir: str | Path) -> Dict[str, np.ndarray]:
    path = Path(ext_dir) / "drug_fingerprints.pkl"
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare external node features.")
    parser.add_argument("--config", default="configs/ablation_hgt.yaml")
    parser.add_argument("--skip-taxonomy", action="store_true")
    parser.add_argument("--skip-mesh", action="store_true")
    parser.add_argument("--skip-drugs", action="store_true")
    args = parser.parse_args()

    from src.config_utils import load_config

    base = Path(__file__).resolve().parents[1]
    cfg = load_config(base / args.config)

    processed_dir = base / cfg["paths"]["processed_dir"]
    ext_cfg = cfg["external_features"]

    ext_dir = base / ext_cfg["cache_dir"]
    ext_dir.mkdir(parents=True, exist_ok=True)

    microbe_ids, disease_ids, drug_ids = collect_unique_ids(processed_dir)
    print(
        f"microbes={len(microbe_ids)} "
        f"diseases={len(disease_ids)} drugs={len(drug_ids)}"
    )

    dims: Dict[str, Any] = {}

    if not args.skip_taxonomy:
        top_k = ext_cfg["taxonomy_top_k"]
        tax_feats, tax_dim = build_taxonomy_features_onehot(microbe_ids, ext_dir, top_k=top_k)
        with open(ext_dir / "taxonomy_features.pkl", "wb") as f:
            pickle.dump(tax_feats, f, protocol=4)
        dims["taxonomy_dim"] = tax_dim

    if not args.skip_mesh:
        mesh_feats, mesh_dim = build_mesh_features(
            disease_ids, ext_dir,
            level1_top_k=int(ext_cfg.get("mesh_level1_top_k", 60)),
            level2_top_k=int(ext_cfg.get("mesh_level2_top_k", 120)),
            level3_top_k=int(ext_cfg.get("mesh_level3_top_k", 200)),
        )
        with open(ext_dir / "mesh_features.pkl", "wb") as f:
            pickle.dump(mesh_feats, f, protocol=4)
        dims["mesh_dim"] = mesh_dim
        dims["mesh_encoding"] = "level0_level1_level2_level3_multihot"
        dims["mesh_level1_top_k"] = int(ext_cfg.get("mesh_level1_top_k", 60))
        dims["mesh_level2_top_k"] = int(ext_cfg.get("mesh_level2_top_k", 120))
        dims["mesh_level3_top_k"] = int(ext_cfg.get("mesh_level3_top_k", 200))

    if not args.skip_drugs:
        fp_dict, fp_dim = build_drug_features(
            drug_ids, ext_dir,
            morgan_nbits=int(ext_cfg["morgan_nbits"]),
            morgan_radius=int(ext_cfg["morgan_radius"]),
        )
        with open(ext_dir / "drug_fingerprints.pkl", "wb") as f:
            pickle.dump(fp_dict, f, protocol=4)
        dims["drug_dim"] = fp_dim
        dims["drug_fp_dim"] = fp_dim

    with open(ext_dir / "feature_dims.json", "w", encoding="utf-8") as f:
        json.dump(dims, f, indent=2)

    print(f"saved to {ext_dir}")
    print(f"dims={dims}")


if __name__ == "__main__":
    main()
