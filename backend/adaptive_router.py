import numpy as np
import networkx as nx

_path_registry = {}  # {tuple(path): {"alpha": int, "beta": int}}

def reset_registry():
    global _path_registry
    _path_registry.clear()

def register_path(path):
    key = tuple(path)
    if key not in _path_registry:
        _path_registry[key] = {"alpha": 1, "beta": 1}

def sample_path_score(path):
    key = tuple(path)
    if key not in _path_registry:
        register_path(path)
    alpha = _path_registry[key]["alpha"]
    beta = _path_registry[key]["beta"]
    score = np.random.beta(alpha, beta)
    return float(score)

def record_outcome(path, was_safe):
    key = tuple(path)
    if key not in _path_registry:
        register_path(path)
    if was_safe:
        _path_registry[key]["alpha"] += 1
    else:
        _path_registry[key]["beta"] += 1

def thompson_select(candidate_paths):
    if not candidate_paths:
        return None
    
    best_path = None
    best_score = -1.0
    
    for path in candidate_paths:
        register_path(path)
        score = sample_path_score(path)
        if score > best_score:
            best_score = score
            best_path = path

    return best_path

def greedy_reroute(G, source, target, flagged_nodes):
    G_clean = G.copy()
    G_clean.remove_nodes_from(flagged_nodes)
    try:
        path = nx.dijkstra_path(G_clean, source, target, weight='latency')
        cost = nx.dijkstra_path_length(G_clean, source, target, weight='latency')
        return {"path": [int(n) for n in path], "cost": float(cost), "found": True, "method": "greedy"}
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return {"path": [], "cost": None, "found": False, "method": "greedy"}

def dp_optimal_reroute(G, source, target, flagged_nodes):
    G_clean = G.copy()
    G_clean.remove_nodes_from(flagged_nodes)
    
    if source not in G_clean or target not in G_clean:
         return {"path": [], "cost": None, "found": False, "method": "dp"}
         
    pred, dist = nx.floyd_warshall_predecessor_and_distance(G_clean, weight='latency')
    try:
        path = nx.reconstruct_path(source, target, pred)
        cost = float(dist[source][target])
        if cost == float('inf'):
            return {"path": [], "cost": None, "found": False, "method": "dp"}
        return {"path": [int(n) for n in path], "cost": cost, "found": True, "method": "dp"}
    except Exception:
        return {"path": [], "cost": None, "found": False, "method": "dp"}

def get_convergence_data():
    data = []
    for key, params in _path_registry.items():
        alpha = params["alpha"]
        beta = params["beta"]
        mean = alpha / (alpha + beta)
        confidence = alpha + beta
        data.append({
            "path_key": str(list(key)),
            "alpha": int(alpha),
            "beta": int(beta),
            "mean": float(mean),
            "confidence": int(confidence)
        })
    data.sort(key=lambda x: x["confidence"], reverse=True)
    return data

def is_path_safe(path, flagged_nodes):
    for node in path:
        if node in flagged_nodes:
            return False
    return True
