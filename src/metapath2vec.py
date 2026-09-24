from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import networkx as nx
from gensim.models import Word2Vec

from typing import Dict

import numpy as np


METAPATHS = [
    ["M", "D", "M"],
    ["M", "Dr", "M"],
    ["D", "Dr", "D"],
    ["M", "D", "Dr", "M"],
    ["M", "Dr", "D", "M"],
    ["D", "M", "D"],

    ["Dr", "M", "Dr"],
    ["Dr", "D", "Dr"],
    ["Dr", "M", "D", "Dr"],
    ["Dr", "D", "M", "Dr"],
]


def node_type(node_id: str) -> str:
    if node_id.startswith("Dr_"):
        return "Dr"
    if node_id.startswith("M_"):
        return "M"
    if node_id.startswith("D_"):
        return "D"
    return node_id.split("_", 1)[0]


def build_typed_adjacency(G: nx.Graph):
    adj_by_type = defaultdict(lambda: defaultdict(list))
    nodes_by_type = defaultdict(list)

    for n in G.nodes():
        nodes_by_type[node_type(n)].append(n)

    for u, v in G.edges():
        tu, tv = node_type(u), node_type(v)
        adj_by_type[u][tv].append(v)
        adj_by_type[v][tu].append(u)

    return adj_by_type, nodes_by_type


def metapath_walk(
    adj_by_type,
    nodes_by_type,
    start_node: str,
    metapath: list[str],
    walk_length: int,
    rng: random.Random,
) -> list[str]:
    walk = [start_node]
    cur = start_node
    pos = 1

    for _ in range(walk_length):
        next_type = metapath[pos]
        candidates = adj_by_type[cur].get(next_type, [])

        if candidates:
            cur = rng.choice(candidates)
        else:
            pool = nodes_by_type.get(next_type, [])
            if not pool:
                break
            cur = rng.choice(pool)

        walk.append(cur)

        pos += 1
        if pos >= len(metapath):
            pos = 1

    return walk


def train_metapath2vec(
    *,
    kg_path: str | Path,
    out_path: str | Path,
    dim: int,
    walk_length: int,
    num_walks: int,
    window: int,
    epochs: int,
    workers: int,
    negative: int,
    min_count: int,
    seed: int,
) -> Path:
    kg_path = Path(kg_path)
    out_path = Path(out_path)
    rng = random.Random(seed)

    G = nx.read_edgelist(str(kg_path), nodetype=str, delimiter="\t", data=False)
    adj_by_type, nodes_by_type = build_typed_adjacency(G)

    walks: list[list[str]] = []
    lengths: list[int] = []

    for mp in METAPATHS:
        starts = nodes_by_type.get(mp[0], [])
        for s in starts:
            for _ in range(num_walks):
                w = metapath_walk(adj_by_type, nodes_by_type, s, mp, walk_length, rng)
                walks.append(w)
                lengths.append(len(w))
    rng.shuffle(walks)

    model = Word2Vec(
        sentences=walks,
        vector_size=dim,
        window=window,
        min_count=min_count,
        sg=1,
        negative=negative,
        workers=workers,
        epochs=epochs,
        seed=seed,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.wv.save_word2vec_format(str(out_path))
    print(f"saved={out_path}")
    return out_path

def load_embeddings_word2vec_txt(path: str) -> Dict[str, np.ndarray]:
    emb: Dict[str, np.ndarray] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().strip().split()
        if len(header) == 2 and header[0].isdigit() and header[1].isdigit():
            dim = int(header[1])
        else:
            parts = header
            token = parts[0]
            vec = np.array(list(map(float, parts[1:])), dtype=np.float32)
            dim = vec.shape[0]
            emb[token] = vec

        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            token = parts[0]
            vec = np.array(list(map(float, parts[1:])), dtype=np.float32)
            if vec.shape[0] != dim:
                continue
            emb[token] = vec
    return emb