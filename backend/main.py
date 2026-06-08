"""
main.py
-------
FastAPI server exposing the RouteGuard anomaly detection layer.

Endpoints
---------
POST /graph/generate          — create & store a new synthetic graph
GET  /graph/summary           — node/edge counts, avg latency, etc.
POST /anomaly/train           — fit Isolation Forest on current graph
POST /anomaly/step            — run one simulation timestep
GET  /anomaly/flagged         — list currently flagged nodes
GET  /anomaly/scores          — all flag scores (0-3) per node
GET  /anomaly/node/{node_id}  — full detail for one node
POST /anomaly/reset           — reset all state, keep graph
POST /anomaly/simulate        — run N timesteps automatically

Run with:
    uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import numpy as np

import graph_utils as gu
import anomaly_detector as ad
import adaptive_router as ar
import routing_engine as re
import simulation as sim

# ── app setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RouteGuard — Anomaly Detection API",
    description=(
        "Three-stage anomaly detection: Z-score | Betweenness Centrality | Isolation Forest"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── shared state ──────────────────────────────────────────────────────────────

_G = None          # current NetworkX DiGraph
_sim_results = None # results from last simulation run
_step_count = 0    # number of timesteps run so far
_step_logs  = []   # history of all step results


# ── request / response models ─────────────────────────────────────────────────

class GraphConfig(BaseModel):
    n_nodes : int = Field(default=20, ge=5,  le=100, description="Number of nodes")
    n_edges : int = Field(default=50, ge=10, le=500, description="Number of edges")
    seed    : int = Field(default=42,                description="Random seed")


class StepRequest(BaseModel):
    inject_spike : Optional[List[int]] = Field(
        default=None,
        description="List of node IDs to artificially spike traffic on (for demo)"
    )
    spike_multiplier: float = Field(
        default=5.0,
        description="How much to multiply normal traffic by on spiked nodes"
    )


class SimulateRequest(BaseModel):
    steps           : int            = Field(default=20, ge=1, le=200)
    inject_spike_at : int            = Field(default=5)
    spike_nodes     : Optional[List[int]] = None
    spike_multiplier: float          = Field(default=8.0)
    source          : int            = Field(default=0)
    target          : int            = Field(default=10)

# ── helper ────────────────────────────────────────────────────────────────────

def _require_graph():
    if _G is None:
        raise HTTPException(status_code=400, detail="No graph loaded. Call POST /graph/generate first.")


def _simulate_traffic(inject_nodes=None, multiplier=5.0) -> Dict[int, float]:
    """Generate one timestep of traffic volumes for all nodes."""
    volumes = {}
    for node in _G.nodes():
        base = float(np.random.normal(loc=100, scale=25))
        base = max(10.0, base)
        if inject_nodes and node in inject_nodes:
            base *= multiplier
        volumes[node] = round(base, 2)
    return volumes


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    return {
        "project" : "RouteGuard",
        "layer"   : "Anomaly Detection (Partial Execution)",
        "status"  : "running",
        "graph_loaded": _G is not None,
        "steps_run"   : _step_count,
    }


# ── Graph endpoints ────────────────────────────────────────────────────────────

@app.post("/graph/generate", tags=["Graph"])
def generate_graph(config: GraphConfig):
    """Generate a new synthetic network graph and reset all detector state."""
    global _G, _step_count, _step_logs
    _G = gu.generate_synthetic_graph(config.n_nodes, config.n_edges, config.seed)
    gu.add_trust_scores(_G)
    ad.reset_state(_G)
    _step_count = 0
    _step_logs  = []
    return {
        "message": "Graph generated and anomaly detector reset.",
        "summary": gu.graph_summary(_G),
        "nodes"  : list(_G.nodes()),
    }


@app.get("/graph/summary", tags=["Graph"])
def graph_summary():
    """Return basic stats about the current graph."""
    _require_graph()
    return gu.graph_summary(_G)


@app.get("/graph/nodes", tags=["Graph"])
def graph_nodes():
    """Return all node IDs and their attributes."""
    _require_graph()
    return {
        str(n): {
            "type"      : _G.nodes[n].get("type"),
            "trust"     : round(_G.nodes[n].get("trust", 1.0), 3),
            "flag_score": _G.nodes[n].get("flag_score", 0),
            "color"     : ad.node_color(n),
        }
        for n in _G.nodes()
    }


# ── Anomaly detection endpoints ────────────────────────────────────────────────

@app.post("/anomaly/train", tags=["Anomaly Detection"])
def train_model():
    """
    Fit the Isolation Forest on the current (normal) state of the graph.
    Must be called before running steps.
    """
    _require_graph()
    msg = ad.train_isolation_forest(_G)
    return {"message": msg}


@app.post("/anomaly/step", tags=["Anomaly Detection"])
def run_step(req: StepRequest = StepRequest()):
    """
    Run ONE simulation timestep.

    What happens inside:
    1. Simulate traffic volumes for all nodes
    2. Stage 1: Z-score spike check
    3. Stage 2: Betweenness centrality delta check
    4. Stage 3: Isolation Forest prediction
    5. Aggregate flag scores → evaluate flagged nodes
    6. Apply latency penalties on flagged nodes

    Returns full per-node anomaly report for this step.
    """
    global _step_count
    _require_graph()

    _step_count += 1
    volumes   = _simulate_traffic(req.inject_spike, req.spike_multiplier)
    results   = ad.update_traffic(_G, volumes)
    penalised = ad.apply_penalties(_G)
    flagged   = ad.get_flagged_nodes()

    # Summarise this step
    flagged_details = {
        node: results[node] for node in flagged if node in results
    }
    watching = [n for n, r in results.items() if r["status"] == "WATCHING"]

    step_summary = {
        "step"            : _step_count,
        "injected_spikes" : req.inject_spike or [],
        "flagged_nodes"   : flagged,
        "watching_nodes"  : watching,
        "penalised_edges" : penalised,
        "node_results"    : results,
        "flagged_details" : flagged_details,
        "summary": {
            "total_nodes"  : _G.number_of_nodes(),
            "normal"       : sum(1 for r in results.values() if r["status"] == "NORMAL"),
            "watching"     : len(watching),
            "flagged"      : len(flagged),
        }
    }

    _step_logs.append({
        "step"         : _step_count,
        "flagged_count": len(flagged),
        "flagged_nodes": flagged,
        "watching"     : watching,
    })

    return step_summary


@app.post("/anomaly/simulate", tags=["Anomaly Detection"])
def simulate(req: SimulateRequest):
    global _step_count, _sim_results
    _require_graph()

    import simulation as sim

    ad.reset_state(_G)
    gu.add_trust_scores(_G)
    ad.train_isolation_forest(_G)
    ar.reset_registry()

    result = sim.run_simulation(
        G               = _G,
        source          = req.source,
        target          = req.target,
        T               = req.steps,
        inject_at       = req.inject_spike_at,
        inject_nodes    = req.spike_nodes or [],
        spike_multiplier= req.spike_multiplier,
    )

    _step_count   = req.steps
    _sim_results  = result
    return result

@app.get("/anomaly/flagged", tags=["Anomaly Detection"])
def get_flagged():
    """Return all currently flagged node IDs with their details."""
    _require_graph()
    flagged = ad.get_flagged_nodes()
    scores  = ad.get_flag_scores()
    return {
        "flagged_nodes": flagged,
        "count"        : len(flagged),
        "details"      : {
            n: {
                "flag_score": scores.get(n, 0),
                "type"      : _G.nodes[n].get("type"),
                "trust"     : round(_G.nodes[n].get("trust", 1.0), 3),
                "color"     : ad.node_color(n),
            }
            for n in flagged
        }
    }


@app.get("/anomaly/scores", tags=["Anomaly Detection"])
def get_all_scores():
    """Return flag scores (0-3) and status for every node."""
    _require_graph()
    scores  = ad.get_flag_scores()
    flagged = set(ad.get_flagged_nodes())
    return {
        str(n): {
            "flag_score": scores.get(n, 0),
            "status"    : ad.node_color(n),
            "color"     : ad.node_color(n),
            "type"      : _G.nodes[n].get("type"),
            "trust"     : round(_G.nodes[n].get("trust", 1.0), 3),
        }
        for n in sorted(_G.nodes())
    }


@app.get("/anomaly/node/{node_id}", tags=["Anomaly Detection"])
def get_node_detail(node_id: int):
    """Return full anomaly detail for a single node."""
    _require_graph()
    if node_id not in _G.nodes():
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found in graph.")

    scores  = ad.get_flag_scores()
    flagged = set(ad.get_flagged_nodes())
    hist    = ad._traffic_history.get(node_id, [])

    return {
        "node_id"       : node_id,
        "type"          : _G.nodes[node_id].get("type"),
        "trust"         : round(_G.nodes[node_id].get("trust", 1.0), 3),
        "flag_score"    : scores.get(node_id, 0),
        "status"        : ad._node_status(node_id),
        "color"         : ad.node_color(node_id),
        "is_flagged"    : node_id in flagged,
        "traffic_history": [round(v, 2) for v in hist],
        "traffic_mean"  : round(float(np.mean(hist)), 2) if hist else None,
        "traffic_std"   : round(float(np.std(hist)), 2)  if hist else None,
        "edges_out"     : [
            {"to": v, "latency": _G.edges[node_id, v]["latency"], "bandwidth": _G.edges[node_id, v]["bandwidth"]}
            for v in _G.successors(node_id)
        ],
    }


@app.get("/anomaly/history", tags=["Anomaly Detection"])
def get_history():
    """Return the timeline log of all steps run so far."""
    return {
        "total_steps": _step_count,
        "history"    : _step_logs,
    }


@app.post("/anomaly/reset", tags=["Anomaly Detection"])
def reset():
    """Reset all anomaly state but keep the current graph."""
    global _step_count, _step_logs
    _require_graph()
    ad.reset_state(_G)
    gu.add_trust_scores(_G)
    _step_count = 0
    _step_logs  = []
    return {"message": "Anomaly state reset. Graph preserved."}


# ── Routing endpoints ─────────────────────────────────────────────────────────

@app.get("/routing/compare", tags=["Routing"])
def compare_routing(source: int, target: int):
    """Run all path-finding algorithms and return comparison."""
    _require_graph()
    if source not in _G or target not in _G:
        raise HTTPException(status_code=400, detail="Source or target not in graph.")
    return re.compare_all_algorithms(_G, source, target)


@app.get("/routing/mst", tags=["Routing"])
def get_mst():
    """Return Minimum Spanning Tree (MST) edges and total cost."""
    _require_graph()
    return re.prim_mst(_G)


@app.get("/routing/maxflow", tags=["Routing"])
def get_max_flow(source: int, sink: int):
    """Return max flow and min cut data."""
    _require_graph()
    if source not in _G or sink not in _G:
        raise HTTPException(status_code=400, detail="Source or sink not in graph.")
    return re.max_flow(_G, source, sink)


@app.get("/routing/topological", tags=["Routing"])
def get_topological_sort():
    """Return topological sort ordering or report if a cycle exists."""
    _require_graph()
    return re.topological_sort(_G)


# ── Adaptive Router endpoints ─────────────────────────────────────────────────

@app.get("/adaptive/convergence", tags=["Adaptive Router"])
def get_convergence():
    """Return Thompson Sampling convergence data per path."""
    return ar.get_convergence_data()


@app.post("/adaptive/reset", tags=["Adaptive Router"])
def reset_adaptive():
    """Clear Thompson Sampling path registry."""
    ar.reset_registry()
    return {"message": "Path registry cleared."}


@app.get("/simulation/results", tags=["Simulation"])
def get_simulation_results():
    """Return _sim_results if available."""
    if _sim_results is None:
        raise HTTPException(status_code=404, detail="No simulation run yet.")
    return _sim_results



