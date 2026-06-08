"""
anomaly_detector.py
-------------------
Implements the three-stage flag scoring system for RouteGuard.

Stage 1 — Z-score thresholding      (traffic volume spike)
Stage 2 — Betweenness centrality    (man-in-the-middle positioning)
Stage 3 — Isolation Forest          (multivariate ML anomaly score)

A node is FLAGGED when flag_score >= 3  (all three stages must agree).
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from collections import defaultdict
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Set, Tuple

# ── module-level state ────────────────────────────────────────────────────────

# Rolling traffic history per node  (last WINDOW readings)
WINDOW = 20
_traffic_history: Dict[int, List[float]] = defaultdict(list)

# Betweenness centrality from the previous step (for delta check)
_prev_centrality: Dict[int, float] = {}

# Flag scores  (0-3)
_flag_scores: Dict[int, int] = defaultdict(int)

# Fully flagged nodes
_flagged_nodes: Set[int] = set()

# Trained Isolation Forest model (fitted once, re-used per step)
_iso_forest: IsolationForest | None = None

# Thresholds (tunable)
ZSCORE_THRESHOLD     = 3
CENTRALITY_DELTA     = 0.10
ISO_CONTAMINATION    = 0.05
FLAG_THRESHOLD       = 2     # ALL three stages must agree to flag a node


# ── public API ────────────────────────────────────────────────────────────────

def reset_state(G: nx.DiGraph) -> None:
    """Clear all state and re-initialise from graph G."""
    global _prev_centrality, _iso_forest
    _traffic_history.clear()
    _flag_scores.clear()
    _flagged_nodes.clear()
    _prev_centrality = {n: 0.0 for n in G.nodes()}
    _iso_forest = None


def train_isolation_forest(G: nx.DiGraph) -> str:
    """
    Fit Isolation Forest on the CURRENT (normal) state of the graph.
    Call this once after graph creation, before the simulation starts.
    """
    global _iso_forest
    X = _build_feature_matrix(G)
    if X is None:
        return "Not enough data to train"
    _iso_forest = IsolationForest(
        contamination=ISO_CONTAMINATION,
        random_state=42,
        n_estimators=100
    )
    _iso_forest.fit(X)
    return f"Isolation Forest trained on {len(X)} nodes"


def update_traffic(G: nx.DiGraph, volumes: Dict[int, float]) -> Dict[int, dict]:
    """
    Feed one timestep of traffic volumes into the detector.

    Parameters
    ----------
    G       : the live network graph
    volumes : {node_id: traffic_volume}  for all nodes this step

    Returns
    -------
    Per-node result dict with z_score, centrality_delta, iso_flag, flag_score, status
    """
    results = {}

    # ── Stage 1: Z-score ──────────────────────────────────────────────────────
    zscore_flags = _check_zscore(volumes)

    # ── Stage 2: Betweenness centrality ───────────────────────────────────────
    centrality_flags, new_centrality = _check_centrality(G)

    # ── Stage 3: Isolation Forest ─────────────────────────────────────────────
    iso_flags = _check_isolation_forest(G)

    # ── Aggregate flag scores ─────────────────────────────────────────────────
    for node in G.nodes():
        prev_score = _flag_scores[node]

        # Increment for each stage triggered this step
        stage1 = zscore_flags.get(node, False)
        stage2 = centrality_flags.get(node, False)
        stage3 = iso_flags.get(node, False)

        triggered = int(stage1) + int(stage2) + int(stage3)

        # Score decays by 1 if no stage fires, grows by triggered count
        if triggered > 0:
            _flag_scores[node] = min(3, prev_score + triggered)
        else:
            _flag_scores[node] = max(0, prev_score - 1)

    # ── Evaluate flags ────────────────────────────────────────────────────────
    _evaluate_flags(G)

    # ── Build per-node result ─────────────────────────────────────────────────
    for node in G.nodes():
        vol  = volumes.get(node, 0.0)
        hist = _traffic_history[node]
        if len(hist) >= 2:
            mean  = float(np.mean(hist[:-1]))
            std   = float(np.std(hist[:-1])) or 1.0
            zscore = round((vol - mean) / std, 3)
        else:
            zscore = 0.0

        c_delta = round(
            new_centrality.get(node, 0.0) - _prev_centrality.get(node, 0.0), 4
        )

        results[node] = {
            "node"              : int(node),
            "node_type"         : G.nodes[node].get("type", "unknown"),
            "traffic_volume"    : float(round(vol, 2)),
            "z_score"           : float(zscore),
            "stage1_zscore"     : bool(zscore_flags.get(node, False)),
            "centrality"        : float(round(new_centrality.get(node, 0.0), 4)),
            "centrality_delta"  : float(c_delta),
            "stage2_centrality" : bool(centrality_flags.get(node, False)),
            "stage3_iso_forest" : bool(iso_flags.get(node, False)),
            "flag_score"        : int(_flag_scores[node]),
            "status"            : _node_status(node),
        }

    # Store new centrality for next step
    _prev_centrality.update(new_centrality)

    return results


def get_flagged_nodes() -> List[int]:
    return [int(n) for n in sorted(_flagged_nodes)]


def get_flag_scores() -> Dict[int, int]:
    return {int(k): int(v) for k, v in _flag_scores.items()}


def apply_penalties(G: nx.DiGraph) -> List[Tuple[int, int]]:
    """
    Set edge latency to PENALTY_LATENCY (fixed cap) for edges adjacent to
    flagged nodes. Restores original latency when a node recovers.
    Never compounds — cost cannot overflow across timesteps.
    """
    PENALTY_LATENCY = 9999
    penalised = []

    for u, v, data in G.edges(data=True):
        u_flagged = u in _flagged_nodes
        v_flagged = v in _flagged_nodes

        if u_flagged or v_flagged:
            if "original_latency" not in data:
                G.edges[u, v]["original_latency"] = data["latency"]
            G.edges[u, v]["latency"] = PENALTY_LATENCY
            penalised.append((int(u), int(v)))
        else:
            if "original_latency" in data:
                G.edges[u, v]["latency"] = data["original_latency"]
                del G.edges[u, v]["original_latency"]

    return penalised


def node_color(node: int) -> str:
    score = _flag_scores.get(node, 0)
    if node in _flagged_nodes:
        return "red"
    if score == 1:
        return "yellow"
    return "green"


# ── internal helpers ──────────────────────────────────────────────────────────

def _check_zscore(volumes: Dict[int, float]) -> Dict[int, bool]:
    flags = {}
    for node, vol in volumes.items():
        _traffic_history[node].append(vol)
        if len(_traffic_history[node]) > WINDOW:
            _traffic_history[node].pop(0)

        hist = _traffic_history[node]
        if len(hist) < 4:
            flags[node] = False
            continue

        mean  = np.mean(hist[:-1])
        std   = np.std(hist[:-1]) or 1.0
        z     = abs((vol - mean) / std)
        flags[node] = bool(z > ZSCORE_THRESHOLD)
    return flags


def _check_centrality(G: nx.DiGraph) -> Tuple[Dict[int, bool], Dict[int, float]]:
    try:
        new_c = nx.betweenness_centrality(G, weight="latency", normalized=True)
    except Exception:
        new_c = {n: 0.0 for n in G.nodes()}

    flags = {}
    for node in G.nodes():
        delta = new_c.get(node, 0.0) - _prev_centrality.get(node, 0.0)
        flags[node] = bool(delta > CENTRALITY_DELTA)
    return flags, new_c


def _check_isolation_forest(G: nx.DiGraph) -> Dict[int, bool]:
    if _iso_forest is None:
        return {n: False for n in G.nodes()}

    X = _build_feature_matrix(G)
    if X is None:
        return {n: False for n in G.nodes()}

    preds = _iso_forest.predict(X)   # -1 = anomaly, 1 = normal
    nodes = sorted(G.nodes())
    return {node: bool(preds[i] == -1) for i, node in enumerate(nodes)}


def _build_feature_matrix(G: nx.DiGraph):
    nodes = sorted(G.nodes())
    if not nodes:
        return None

    try:
        centrality = nx.betweenness_centrality(G, weight="latency", normalized=True)
    except Exception:
        centrality = {n: 0.0 for n in nodes}

    rows = []
    for node in nodes:
        degree     = G.degree(node)
        edges      = list(G.out_edges(node, data=True)) + list(G.in_edges(node, data=True))
        latencies  = [d.get("latency", 50) for _, _, d in edges] or [50]
        hist       = _traffic_history.get(node, [100.0])
        rows.append([
            degree,
            np.mean(latencies),
            np.mean(hist),
            np.std(hist) if len(hist) > 1 else 0.0,
            centrality.get(node, 0.0),
        ])

    return np.array(rows, dtype=float)


def _evaluate_flags(G: nx.DiGraph) -> None:
    for node in list(G.nodes()):
        if _flag_scores[node] >= FLAG_THRESHOLD:
            _flagged_nodes.add(node)
            G.nodes[node]["trust"] = max(0.0, G.nodes[node].get("trust", 1.0) - 0.2)
        elif node in _flagged_nodes and _flag_scores[node] == 0:
            _flagged_nodes.discard(node)
            G.nodes[node]["trust"] = 1.0


def _node_status(node: int) -> str:
    score = _flag_scores.get(node, 0)
    if node in _flagged_nodes:
        return "FLAGGED"
    if score == 1:
        return "WATCHING"
    return "NORMAL"
