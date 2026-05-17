"""
graph_utils.py
--------------
Generates and manages the synthetic network graph used across RouteGuard.
Every node and edge carries the attributes that the anomaly detector reads.
"""

import random
import networkx as nx
import numpy as np
import pandas as pd


def generate_synthetic_graph(n_nodes: int = 30, n_edges: int = 80, seed: int = 42) -> nx.DiGraph:
    """
    Build a random directed graph simulating a network topology.

    Node attributes
    ---------------
    type        : one of router / server / endpoint / relay
    trust       : float 1.0  (degrades when anomalies are detected)
    flag_score  : int   0    (incremented by anomaly detector stages)

    Edge attributes
    ---------------
    latency     : int  1–100  ms   (primary routing weight)
    bandwidth   : int  10–1000 Mbps (used by max-flow)
    trust       : float 1.0
    """
    random.seed(seed)
    np.random.seed(seed)

    G_undirected = nx.gnm_random_graph(n_nodes, n_edges, seed=seed)
    G = nx.DiGraph()
    G.add_nodes_from(G_undirected.nodes())

    node_types = ["router", "server", "endpoint", "relay"]
    for node in G.nodes():
        G.nodes[node]["type"]       = random.choice(node_types)
        G.nodes[node]["trust"]      = 1.0
        G.nodes[node]["flag_score"] = 0

    for u, v in G_undirected.edges():
        latency   = random.randint(1, 100)
        bandwidth = random.randint(10, 1000)
        G.add_edge(u, v, latency=latency, bandwidth=bandwidth, trust=1.0)
        # Add reverse edge with slightly different weights to make it directed
        G.add_edge(v, u,
                   latency=random.randint(1, 100),
                   bandwidth=random.randint(10, 1000),
                   trust=1.0)

    return G


def add_trust_scores(G: nx.DiGraph) -> None:
    """Reset / initialise trust scores on all nodes and edges."""
    for node in G.nodes():
        G.nodes[node]["trust"]      = 1.0
        G.nodes[node]["flag_score"] = 0
    for u, v in G.edges():
        G.edges[u, v]["trust"] = 1.0


def graph_summary(G: nx.DiGraph) -> dict:
    types, counts = np.unique([G.nodes[n]["type"] for n in G.nodes()], return_counts=True)
    return {
        "nodes": int(G.number_of_nodes()),
        "edges": int(G.number_of_edges()),
        "node_types": {str(t): int(c) for t, c in zip(types, counts)},
        "avg_latency": float(round(np.mean([G.edges[e]["latency"] for e in G.edges()]), 2)),
        "avg_bandwidth": float(round(np.mean([G.edges[e]["bandwidth"] for e in G.edges()]), 2)),
    }


def load_csv_graph(filepath: str) -> nx.DiGraph:
    """
    Read a CSV with columns: source, target, latency, bandwidth
    Build and return a DiGraph with those edges.
    Each node gets default attributes: type='router', trust=1.0, flag_score=0
    """
    df = pd.read_csv(filepath)
    G = nx.DiGraph()
    
    for _, row in df.iterrows():
        G.add_edge(
            int(row["source"]), 
            int(row["target"]),
            latency=int(row["latency"]),
            bandwidth=int(row["bandwidth"]),
            trust=1.0
        )
        
    for node in G.nodes():
        G.nodes[node]["type"] = "router"
        G.nodes[node]["trust"] = 1.0
        G.nodes[node]["flag_score"] = 0
        
    return G
