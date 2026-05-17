# RouteGuard — Adaptive Network Routing with Anomaly-Aware Path Optimization
## Complete Project Specification for Agentic IDE

---

## 1. PROJECT OVERVIEW

RouteGuard is a DAA (Design and Analysis of Algorithms) project that simulates a
self-healing, security-aware network. It models a computer network as a weighted
directed graph and does three things simultaneously:

1. **Routes** packets efficiently using classical shortest-path algorithms
2. **Detects** suspicious/compromised nodes using graph algorithms and ML
3. **Adapts** future routing decisions using Thompson Sampling (Multi-Armed Bandit)

The system exposes a FastAPI backend. The frontend is a Streamlit dashboard.

### The core loop every timestep:
```
Generate traffic → Detect anomalies → Flag bad nodes →
Penalize bad edges → Route around them → Learn from outcome → Repeat
```

---

## 2. TECH STACK

| Component        | Technology                          |
|------------------|-------------------------------------|
| Language         | Python 3.11                         |
| Backend API      | FastAPI + Uvicorn                   |
| Frontend         | Streamlit                           |
| Graph engine     | NetworkX                            |
| ML               | scikit-learn (Isolation Forest)     |
| Data             | NumPy, Pandas                       |
| Visualization    | Plotly, Matplotlib                  |
| HTTP client      | httpx (Streamlit → FastAPI calls)   |

---

## 3. FOLDER STRUCTURE

Create exactly this structure:

```
RouteGuard/
├── backend/
│   ├── main.py                  # FastAPI app — all API endpoints
│   ├── graph_utils.py           # Graph generation and management
│   ├── anomaly_detector.py      # Three-stage anomaly detection engine
│   ├── routing_engine.py        # All shortest-path and graph algorithms
│   ├── adaptive_router.py       # Thompson Sampling + rerouting
│   └── simulation.py            # Full simulation loop tying all layers
├── frontend/
│   └── app.py                   # Streamlit dashboard — 5 tabs
├── data/
│   └── sample_graph.csv         # Sample edge list for demo
├── requirements.txt
└── README.md
```

---

## 4. REQUIREMENTS.TXT

```
fastapi==0.111.0
uvicorn==0.29.0
networkx==3.3
scikit-learn==1.4.2
numpy==1.26.4
pandas==2.2.2
httpx==0.27.0
pydantic==2.7.1
streamlit==1.35.0
plotly==5.22.0
matplotlib==3.9.0
anyio==4.3.0
```

---

## 5. FILE-BY-FILE SPECIFICATION

---

### FILE 1: backend/graph_utils.py

**Purpose:** Generate and manage the network graph. Every other module imports from here.

**What it must do:**

#### Function: `generate_synthetic_graph(n_nodes, n_edges, seed)`
- Use `networkx.gnm_random_graph(n_nodes, n_edges, seed=seed)` to create an
  undirected base graph, then convert it to a `networkx.DiGraph`
- For every node, assign these attributes:
  - `type`: randomly chosen from `["router", "server", "endpoint", "relay"]`
  - `trust`: float, starts at `1.0`
  - `flag_score`: int, starts at `0`
- For every edge (u→v), assign:
  - `latency`: random int between 1 and 100 (milliseconds, used as routing weight)
  - `bandwidth`: random int between 10 and 1000 (Mbps, used for max-flow)
  - `trust`: float, starts at `1.0`
- Also add reverse edges (v→u) with different random latency/bandwidth values
  to make it a proper directed graph
- Return the DiGraph

#### Function: `add_trust_scores(G)`
- Reset all node attributes: `trust=1.0`, `flag_score=0`
- Reset all edge attributes: `trust=1.0`
- Modifies G in place, returns nothing

#### Function: `graph_summary(G)`
- Return a plain Python dict (no numpy types) with:
  - `nodes`: int count of nodes
  - `edges`: int count of edges
  - `node_types`: dict mapping type string → int count
  - `avg_latency`: float average latency across all edges
  - `avg_bandwidth`: float average bandwidth across all edges
- IMPORTANT: cast all numpy types to Python native types (int, float, str)
  to avoid FastAPI JSON serialization errors

#### Function: `load_csv_graph(filepath)`
- Read a CSV with columns: `source, target, latency, bandwidth`
- Build and return a DiGraph with those edges
- Each node gets default attributes: `type="router"`, `trust=1.0`, `flag_score=0`

---

### FILE 2: backend/anomaly_detector.py

**Purpose:** Three-stage anomaly detection engine. Maintains module-level state
across simulation timesteps.

**Module-level state variables (declare at top of file):**
```python
WINDOW = 20                    # rolling traffic history size
ZSCORE_THRESHOLD = 2.5         # z-score spike threshold
CENTRALITY_DELTA = 0.10        # betweenness centrality spike threshold
ISO_CONTAMINATION = 0.05       # Isolation Forest contamination rate
FLAG_THRESHOLD = 2             # minimum stages to fully flag a node

_traffic_history = defaultdict(list)   # node → list of last 20 volumes
_prev_centrality = {}                  # node → float centrality from last step
_flag_scores = defaultdict(int)        # node → int 0-3
_flagged_nodes = set()                 # set of fully flagged node IDs
_iso_forest = None                     # trained IsolationForest model
```

#### Function: `reset_state(G)`
- Clear `_traffic_history`, `_flag_scores`, `_flagged_nodes`
- Set `_prev_centrality = {n: 0.0 for n in G.nodes()}`
- Set `_iso_forest = None`

#### Function: `train_isolation_forest(G)`
- Build a feature matrix X where each row = one node, columns =
  `[degree, avg_latency_of_edges, traffic_mean, traffic_std, betweenness_centrality]`
- Compute betweenness centrality using `networkx.betweenness_centrality(G, weight='latency')`
- For nodes with no traffic history yet, use default values: traffic_mean=100, traffic_std=0
- Fit `IsolationForest(contamination=ISO_CONTAMINATION, random_state=42, n_estimators=100)`
- Store fitted model in `_iso_forest`
- Return a string message like "Isolation Forest trained on N nodes"

#### Function: `update_traffic(G, volumes)`
- `volumes` is a dict: `{node_id: float_volume}`
- This is the main function called every simulation timestep
- It runs all three stages and returns per-node results

**Inside update_traffic, run these stages in order:**

**Stage 1 — Z-score spike check:**
- For each node, append its volume to `_traffic_history[node]`
- Trim history to last WINDOW readings
- If history has fewer than 4 readings, skip (not enough data yet)
- Compute `mean` and `std` of history EXCLUDING the latest value
- `z_score = abs((latest_volume - mean) / std)`
- If `z_score > ZSCORE_THRESHOLD`: this stage fires for this node → `stage1 = True`

**Stage 2 — Betweenness centrality spike check:**
- Compute `new_centrality = networkx.betweenness_centrality(G, weight='latency', normalized=True)`
- For each node: `delta = new_centrality[node] - _prev_centrality[node]`
- If `delta > CENTRALITY_DELTA`: this stage fires → `stage2 = True`
- After checking all nodes, update `_prev_centrality = new_centrality`

**Stage 3 — Isolation Forest:**
- If `_iso_forest` is None: skip (not trained yet), all nodes get `stage3 = False`
- Build the same feature matrix as in `train_isolation_forest`
- Call `_iso_forest.predict(X)` — returns array of 1 (normal) or -1 (anomaly)
- If a node gets -1: `stage3 = True`

**Flag score update:**
- For each node: count how many stages fired (`triggered = int(stage1) + int(stage2) + int(stage3)`)
- If `triggered > 0`: increment `_flag_scores[node]` by `triggered`, cap at 3
- If `triggered == 0`: decrement `_flag_scores[node]` by 1, floor at 0
  (this allows nodes to recover over time)

**Evaluate flags:**
- For each node:
  - If `_flag_scores[node] >= FLAG_THRESHOLD`: add to `_flagged_nodes`,
    reduce `G.nodes[node]['trust']` by 0.2 (floor at 0.0)
  - If `_flag_scores[node] == 0` and node in `_flagged_nodes`:
    remove from `_flagged_nodes`, restore `G.nodes[node]['trust'] = 1.0`

**Build and return results dict:**
Return `{node_id: {details}}` for every node. Each node dict must contain:
```python
{
    "node": int(node),
    "node_type": str,
    "traffic_volume": float,
    "z_score": float,
    "stage1_zscore": bool,
    "centrality": float,
    "centrality_delta": float,
    "stage2_centrality": bool,
    "stage3_iso_forest": bool,
    "flag_score": int,
    "status": "NORMAL" | "WATCHING" | "FLAGGED"
}
```
Status rules: FLAGGED if in `_flagged_nodes`, WATCHING if `flag_score == 1`, else NORMAL

IMPORTANT: Cast all numpy types to Python native types before returning.

#### Function: `get_flagged_nodes()`
- Return sorted list of flagged node IDs as Python ints

#### Function: `get_flag_scores()`
- Return dict of `{int(node): int(score)}` for all nodes

#### Function: `apply_penalties(G)`
- For every node in `_flagged_nodes`:
  - Get all in_edges and out_edges of that node
  - Multiply `G.edges[u,v]['latency']` by 10 for each adjacent edge
- Return list of `(int(u), int(v))` tuples for all penalized edges
- NOTE: This permanently modifies G. The simulation resets G when needed.

#### Function: `node_color(node)`
- Return "red" if node in `_flagged_nodes`
- Return "yellow" if `_flag_scores[node] == 1`
- Return "green" otherwise

#### Function: `node_status(node)`
- Return "FLAGGED", "WATCHING", or "NORMAL" using same logic as above

---

### FILE 3: backend/routing_engine.py

**Purpose:** Implement all graph/routing DAA algorithms. Each algorithm function
takes graph G, source node, target node, and returns a result dict.

**Standard return format for path-finding functions:**
```python
{
    "algorithm": str,
    "path": list of int node IDs,
    "cost": float (total latency),
    "hops": int (number of edges),
    "time_ms": float (wall-clock execution time in milliseconds),
    "found": bool (False if no path exists)
}
```

#### Function: `dijkstra(G, source, target)`
- Use `networkx.dijkstra_path(G, source, target, weight='latency')`
- Use `networkx.dijkstra_path_length(G, source, target, weight='latency')` for cost
- Measure wall-clock time using `time.perf_counter()` before and after
- Return standard result dict with algorithm="Dijkstra"
- If no path exists (NetworkXNoPath), return found=False, empty path, cost=inf

#### Function: `bellman_ford(G, source, target)`
- Use `networkx.bellman_ford_path(G, source, target, weight='latency')`
- Use `networkx.bellman_ford_path_length(G, source, target, weight='latency')` for cost
- Catch NetworkXUnbounded (negative cycle) — return found=False if caught
- Return standard result dict with algorithm="Bellman-Ford"

#### Function: `astar(G, source, target)`
- Use `networkx.astar_path(G, source, target, weight='latency')`
- Heuristic function: `lambda u, v: abs(u - v) * 0.5`
- Compute path cost manually: sum of `G.edges[u,v]['latency']` for consecutive node pairs
- Return standard result dict with algorithm="A*"

#### Function: `bfs_path(G, source, target)`
- Implement BFS manually using `collections.deque`
- Track visited nodes with a set
- Track parent map to reconstruct path
- Cost = sum of latencies along found path
- Return standard result dict with algorithm="BFS"

#### Function: `dfs_path(G, source, target)`
- Implement DFS manually using a stack (list)
- Track visited nodes
- Return standard result dict with algorithm="DFS"

#### Function: `floyd_warshall(G)`
- Use `networkx.floyd_warshall_predecessor_and_distance(G, weight='latency')`
- Returns `(predecessor_dict, distance_dict)` — store both for path reconstruction
- Return dict: `{"distance_matrix": nested dict, "predecessor": nested dict}`
- This is used by dp_optimal_reroute in adaptive_router.py

#### Function: `prim_mst(G)`
- Convert G to undirected: `G.to_undirected()`
- Run `networkx.minimum_spanning_tree(G_undirected, weight='latency')`
- Return dict with: `{"edges": list of (u,v,latency) tuples, "total_cost": float, "node_count": int}`

#### Function: `max_flow(G, source, sink)`
- Use `networkx.maximum_flow(G, source, sink, capacity='bandwidth')`
- Use `networkx.minimum_cut(G, source, sink, capacity='bandwidth')`
- Return dict: `{"flow_value": float, "cut_value": float, "reachable": list, "non_reachable": list}`
- If source==sink or no path: return flow_value=0

#### Function: `topological_sort(G)`
- Try `list(networkx.topological_sort(G))`
- If NetworkXUnfeasible (cycle exists): return `{"success": False, "cycle_detected": True, "order": []}`
- If success: return `{"success": True, "cycle_detected": False, "order": list_of_nodes}`

#### Function: `compare_all_algorithms(G, source, target)`
- Call dijkstra, bellman_ford, astar, bfs_path, dfs_path on the same G, source, target
- Collect all results into a list
- Return: `{"results": list_of_result_dicts, "fastest_algorithm": name_of_lowest_time_ms}`

---

### FILE 4: backend/adaptive_router.py

**Purpose:** Thompson Sampling over candidate paths + greedy/DP rerouting.

**Module-level state:**
```python
_path_registry = {}   # {tuple(path): {"alpha": int, "beta": int}}
```

#### Function: `reset_registry()`
- Clear `_path_registry`

#### Function: `register_path(path)`
- Convert path to tuple: `key = tuple(path)`
- If key not in `_path_registry`: set `_path_registry[key] = {"alpha": 1, "beta": 1}`

#### Function: `sample_path_score(path)`
- key = tuple(path)
- If not registered, register it first
- Sample from Beta distribution: `numpy.random.beta(alpha, beta)`
- Return float score between 0 and 1

#### Function: `record_outcome(path, was_safe)`
- key = tuple(path)
- Register if not already registered
- If `was_safe=True`: increment alpha by 1
- If `was_safe=False`: increment beta by 1

#### Function: `thompson_select(candidate_paths)`
- `candidate_paths` is a list of paths (each path is a list of node IDs)
- For each path, register it and sample its Beta score
- Return the path with the highest sampled score
- If list is empty: return None

#### Function: `greedy_reroute(G, source, target, flagged_nodes)`
- Create a copy of G: `G_clean = G.copy()`
- Remove flagged nodes: `G_clean.remove_nodes_from(flagged_nodes)`
- Try `networkx.dijkstra_path(G_clean, source, target, weight='latency')`
- If no path: return `{"path": [], "cost": None, "found": False, "method": "greedy"}`
- Return `{"path": path, "cost": total_latency, "found": True, "method": "greedy"}`

#### Function: `dp_optimal_reroute(G, source, target, flagged_nodes)`
- Create G_clean with flagged nodes removed
- Run `networkx.floyd_warshall_predecessor_and_distance(G_clean, weight='latency')`
- Reconstruct path using `networkx.reconstruct_path(source, target, predecessor)`
- If source or target removed, or no path: return found=False
- Return `{"path": path, "cost": float, "found": True, "method": "dp"}`

#### Function: `get_convergence_data()`
- For each path in `_path_registry`:
  - Compute `mean = alpha / (alpha + beta)`
  - Compute `confidence = alpha + beta` (total observations)
- Return list of dicts: `[{"path_key": str, "alpha": int, "beta": int, "mean": float, "confidence": int}]`
- Sort by confidence descending

#### Function: `is_path_safe(path, flagged_nodes)`
- Return True if none of the nodes in path are in flagged_nodes
- Return False otherwise

---

### FILE 5: backend/simulation.py

**Purpose:** Ties all three layers together into one simulation loop.

**Function: `run_simulation(G, source, target, T, inject_at, inject_nodes, spike_multiplier)`**

Parameters:
- `G`: NetworkX DiGraph
- `source`: int source node ID
- `target`: int target node ID
- `T`: int number of timesteps to run
- `inject_at`: int timestep number at which to inject a traffic spike
- `inject_nodes`: list of node IDs to spike
- `spike_multiplier`: float how much to multiply normal traffic on spiked nodes

For each timestep t from 1 to T:

**Step 1 — Generate traffic volumes:**
- For each node: `volume = max(10, numpy.random.normal(100, 25))`
- If this is the inject timestep and node is in inject_nodes: `volume *= spike_multiplier`

**Step 2 — Run anomaly detection:**
- Call `anomaly_detector.update_traffic(G, volumes)` → `node_results`

**Step 3 — Apply penalties:**
- Call `anomaly_detector.apply_penalties(G)`

**Step 4 — Get current flagged nodes:**
- Call `anomaly_detector.get_flagged_nodes()` → `flagged`

**Step 5 — Run routing algorithms:**
- Call `routing_engine.compare_all_algorithms(G, source, target)` → `algo_results`

**Step 6 — Thompson Sampling selection:**
- Extract all found paths from `algo_results["results"]`
- Call `adaptive_router.thompson_select(candidate_paths)` → `best_path`

**Step 7 — Record outcome:**
- Check if best_path is safe: `adaptive_router.is_path_safe(best_path, flagged)`
- Call `adaptive_router.record_outcome(best_path, was_safe)`

**Step 8 — Reroute if needed:**
- If not safe:
  - Call `adaptive_router.greedy_reroute(G, source, target, flagged)` → `greedy_result`
  - Call `adaptive_router.dp_optimal_reroute(G, source, target, flagged)` → `dp_result`
- Else: `greedy_result = None`, `dp_result = None`

**Step 9 — MST every 5 steps:**
- If `t % 5 == 0`: call `routing_engine.prim_mst(G)` → `mst_result`
- Else: `mst_result = None`

**Step 10 — Max-flow every 10 steps:**
- If `t % 10 == 0`: call `routing_engine.max_flow(G, source, target)` → `flow_result`
- Else: `flow_result = None`

**Step 11 — Topological sort every 5 steps:**
- If `t % 5 == 0`: call `routing_engine.topological_sort(G)` → `topo_result`
- Else: `topo_result = None`

**Store per-timestep log dict:**
```python
{
    "step": t,
    "spike_injected": (t == inject_at),
    "volumes": {int(k): float(v) for k,v in volumes.items()},
    "flagged_nodes": flagged,
    "watching_nodes": [n for n,r in node_results.items() if r["status"]=="WATCHING"],
    "flagged_count": len(flagged),
    "algo_results": algo_results,
    "best_path": best_path,
    "was_safe": was_safe,
    "greedy_reroute": greedy_result,
    "dp_reroute": dp_result,
    "mst": mst_result,
    "max_flow": flow_result,
    "topo_sort": topo_result,
    "node_results": node_results,
}
```

**Return final dict:**
```python
{
    "steps_run": T,
    "source": source,
    "target": target,
    "timeline": list_of_step_logs,
    "final_flagged": anomaly_detector.get_flagged_nodes(),
    "final_scores": anomaly_detector.get_flag_scores(),
    "convergence": adaptive_router.get_convergence_data(),
    "final_node_colors": {int(n): anomaly_detector.node_color(n) for n in G.nodes()},
}
```

---

### FILE 6: backend/main.py

**Purpose:** FastAPI app exposing all functionality as REST endpoints.

**Shared state at module level:**
```python
_G = None           # current graph
_sim_results = None # results from last simulation run
```

**All endpoints:**

```
GET  /                          → health check, returns graph_loaded bool, steps run
POST /graph/generate            → body: {n_nodes, n_edges, seed} → generate graph, reset all state
GET  /graph/summary             → returns graph_summary(G)
GET  /graph/nodes               → returns all nodes with attributes and color
POST /anomaly/train             → fit Isolation Forest on current graph
POST /anomaly/step              → body: {inject_spike?: list[int], spike_multiplier?: float}
                                  → run ONE timestep, return full node_results
POST /anomaly/simulate          → body: {steps, inject_spike_at, spike_nodes, spike_multiplier}
                                  → run full simulation, store in _sim_results, return timeline
GET  /anomaly/flagged           → list flagged nodes with details
GET  /anomaly/scores            → all flag scores per node
GET  /anomaly/node/{node_id}    → deep detail for one node including traffic history
GET  /anomaly/history           → full timeline log of all steps run
POST /anomaly/reset             → reset detector state, keep graph
GET  /routing/compare           → query params: source, target → run all algorithms, return comparison
GET  /routing/mst               → return MST edges and total cost
GET  /routing/maxflow           → query params: source, sink → return flow value and min-cut
GET  /routing/topological       → return topological sort or cycle detection
GET  /adaptive/convergence      → return Thompson Sampling convergence data per path
POST /adaptive/reset            → clear path registry
GET  /simulation/results        → return _sim_results if available
```

**CORS:** Enable CORS for all origins (needed for Streamlit frontend to call FastAPI).

**Error handling:** Always check if G is loaded before running any algorithm.
Return HTTP 400 with message "No graph loaded. Call POST /graph/generate first."

**JSON serialization:** Always cast numpy types to Python native before returning.
Add a custom JSON encoder or convert in each endpoint function.

---

### FILE 7: frontend/app.py

**Purpose:** Streamlit dashboard that calls the FastAPI backend.

**Configuration:**
```python
st.set_page_config(page_title="RouteGuard", layout="wide")
API_BASE = "http://localhost:8000"
```

**Sidebar controls:**
- Number input: n_nodes (default 20, min 5, max 100)
- Number input: n_edges (default 50, min 10, max 300)
- Number input: seed (default 42)
- Button: "Generate Graph" → POST /graph/generate → POST /anomaly/train
- Divider
- Number input: source node (default 0)
- Number input: target node (default 10)
- Number input: simulation steps T (default 20)
- Number input: inject spike at step (default 5)
- Text input: spike nodes (comma-separated, e.g. "3,7,12")
- Number input: spike multiplier (default 6.0)
- Button: "Run Simulation" → POST /anomaly/simulate → store results in session_state

**5 Tabs:**

#### Tab 1 — Network Map
- Call GET /graph/nodes → get all node positions and colors
- Use `networkx.spring_layout(G)` with same seed to get node positions
- Draw with Plotly scatter + lines:
  - Nodes: colored by status (green=NORMAL, yellow=WATCHING, red=FLAGGED)
  - Edges: grey lines
  - Hover shows: node id, type, trust, flag_score, status
- If simulation ran: highlight the best_path edges in orange
- Show legend: green=Normal, yellow=Watching, red=Flagged

#### Tab 2 — Routing Analysis
- Call GET /routing/compare?source={source}&target={target}
- Show results as st.dataframe with columns: algorithm, path, cost, hops, time_ms
- Plotly bar chart: algorithm vs cost
- Plotly bar chart: algorithm vs time_ms
- Show "Fastest algorithm: X" and "Cheapest path: X"
- Display the actual path node sequence for each algorithm

#### Tab 3 — Anomaly Report
- Call GET /anomaly/scores → show table of all nodes
- Color rows: red if FLAGGED, yellow if WATCHING, green if NORMAL
- Show columns: node, type, flag_score, status, trust
- If simulation ran: show line chart of flagged_count over simulation timesteps
- Show list of currently flagged nodes with their flag_score breakdown

#### Tab 4 — Rerouting Log
- Requires simulation to have run
- For each timestep where a reroute happened (was_safe=False):
  - Show: timestep, original best_path, flagged nodes, greedy reroute result, dp reroute result
  - Show cost comparison: original cost vs rerouted cost
- Summary: how many reroutes happened total, avg cost saving

#### Tab 5 — Algorithm Comparison
- Show hardcoded complexity table:
  | Algorithm | Category | Complexity | Role in RouteGuard |
  |---|---|---|---|
  | Dijkstra | Graph | O((V+E) log V) | Primary routing |
  | Bellman-Ford | Graph | O(VE) | Penalty-aware routing |
  | A* | Graph | O(E log V) | Heuristic routing |
  | BFS | Traversal | O(V+E) | Hop-minimal path + alert spread |
  | DFS | Traversal | O(V+E) | Path auditing |
  | Floyd-Warshall | DP | O(V³) | DP rerouting |
  | Prim's MST | Greedy | O(E log V) | Network backbone |
  | Max-Flow | Flow | O(VE²) | Bottleneck detection |
  | Betweenness Centrality | Graph | O(VE) | Anomaly detection |
  | Union-Find | DSU | O(α(V)) | Cluster tracking |
  | Topological Sort | DAG | O(V+E) | Packet ordering |
  | Isolation Forest | ML | O(n log n) | Multivariate anomaly |
  | Thompson Sampling | Probabilistic | O(1) | Adaptive path selection |

- If simulation ran: show Plotly line chart of Thompson Sampling convergence
  (x=path_key, y=mean score alpha/(alpha+beta))
- If simulation ran: show regret curve — cumulative cost of best_path vs
  random baseline path (pick random path from algo results each step)

---

## 6. HOW THE THREE LAYERS CONNECT

```
graph_utils.py
      ↓ (provides G)
anomaly_detector.py          routing_engine.py
      ↓ (flags nodes)               ↓ (finds candidate paths)
      ↓ (penalizes edges) ──────────↓
      └────────→ adaptive_router.py ←─────────┘
                      ↓ (Thompson Sampling picks best path)
                      ↓ (records outcome → updates Beta distributions)
                 simulation.py
                      ↓ (runs T timesteps, collects all logs)
                 main.py (FastAPI)
                      ↓ (exposes everything as REST endpoints)
                 frontend/app.py (Streamlit)
                      ↓ (calls API, renders dashboard)
```

---

## 7. DATA FLOW DETAIL

### What happens in ONE simulation timestep:

```
1. numpy.random.normal(100, 25) per node → traffic volumes dict

2. anomaly_detector.update_traffic(G, volumes):
   Stage 1: Z-score = |volume - rolling_mean| / rolling_std
            if Z > 2.5 → stage1_fires = True
   Stage 2: betweenness_centrality(G) → delta from previous step
            if delta > 0.10 → stage2_fires = True
   Stage 3: IsolationForest.predict(feature_matrix)
            if predict == -1 → stage3_fires = True
   flag_score += (stage1 + stage2 + stage3)
   if flag_score >= 2 → node FLAGGED, trust -= 0.2

3. anomaly_detector.apply_penalties(G):
   For flagged nodes: adjacent edge latency *= 10

4. routing_engine.compare_all_algorithms(G, source, target):
   Dijkstra, Bellman-Ford, A*, BFS, DFS all run on penalized G
   Flagged nodes naturally avoided because their edges are expensive

5. adaptive_router.thompson_select(candidate_paths):
   For each candidate path: sample Beta(alpha, beta)
   Pick path with highest sample

6. adaptive_router.is_path_safe(best_path, flagged_nodes):
   True if no flagged node on path

7. adaptive_router.record_outcome(best_path, was_safe):
   was_safe=True  → alpha += 1  (path trust grows)
   was_safe=False → beta  += 1  (path distrust grows)
   → triggers rerouting via greedy and DP
```

---

## 8. KEY ALGORITHM EXPLANATIONS

### Why Dijkstra is the baseline:
Dijkstra finds the minimum latency path in O((V+E) log V). It is the industry
standard (OSPF protocol uses it). RouteGuard uses it as the primary routing
algorithm and benchmarks all others against it.

### Why Bellman-Ford is needed:
When flagged nodes get latency × 10 penalties, edge weights become very high.
Bellman-Ford handles extreme weight variations better and also detects negative
cycles (which would indicate a routing loop attack). Runs in O(VE).

### Why A* is faster on large graphs:
A* uses a heuristic `h(u,v) = |u-v| * 0.5` to guide search toward the target,
exploring fewer nodes than Dijkstra. On 50+ node graphs it is measurably faster.

### Why betweenness centrality detects attacks:
A compromised node being used as a man-in-the-middle relay will suddenly appear
on many more shortest paths than before. Its betweenness centrality spikes.
This is the Stage 2 detection signal.

### Why Isolation Forest needs no labeled data:
Real attack data is rare and hard to label. Isolation Forest is unsupervised —
it learns what "normal" nodes look like from the initial graph state, then flags
nodes that deviate from that normal baseline. Anomalies are isolated faster in
random trees (fewer splits needed), hence "isolation" forest.

### Why Thompson Sampling over Dijkstra alone:
Dijkstra only knows current edge weights. Thompson Sampling knows the history
of every path — which paths have been consistently safe over many timesteps.
Even if a path looks cheap right now (low latency), Thompson Sampling will
deprioritize it if it has historically contained flagged nodes (high beta value).
This is the core novelty: security history encoded in Beta distributions.

### Why compare greedy vs DP rerouting:
Greedy (Dijkstra on cleaned graph) is O(E log V) — fast but locally optimal.
DP (Floyd-Warshall on cleaned graph) is O(V³) — slow but globally optimal.
Showing when they agree and when they differ is the core algorithm analysis
required for a DAA project: empirical proof of the complexity-optimality tradeoff.

---

## 9. SIMULATION DEMO SCENARIO

Use these exact parameters for a clean demo:
- n_nodes = 20, n_edges = 50, seed = 42
- source = 0, target = 15
- T = 20 timesteps
- inject_at = 5, spike_nodes = [7, 12], spike_multiplier = 8.0

Expected behavior:
- Steps 1-4: All nodes green, routing finds direct paths
- Step 5: Nodes 7 and 12 get traffic spike → Z-score fires
- Step 6-7: Centrality delta fires on 7 and 12 → flag_score reaches 2 → FLAGGED (red)
- Routing algorithms see 10x latency penalty on their edges → route around them
- Thompson Sampling records unsafe outcomes → beta rises for paths through 7,12
- Steps 8-15: System stabilizes on safe paths, Thompson Sampling alpha rises
- Step 16+: Traffic normalizes on 7,12 → flag_score decays → nodes recover to green
- Thompson Sampling cautiously re-explores paths through recovered nodes

---

## 10. HOW TO RUN THE PROJECT

```bash
# 1. Create venv with Python 3.11
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the FastAPI backend (terminal 1)
cd backend
uvicorn main:app --reload --port 8000

# 4. Start the Streamlit frontend (terminal 2)
cd frontend
streamlit run app.py

# 5. Open browser
# FastAPI interactive docs: http://localhost:8000/docs
# Streamlit dashboard:      http://localhost:8501
```

---

## 11. WHAT THE AGENTIC IDE MUST BUILD — CHECKLIST

- [ ] `backend/graph_utils.py` — generate_synthetic_graph, add_trust_scores, graph_summary, load_csv_graph
- [ ] `backend/anomaly_detector.py` — reset_state, train_isolation_forest, update_traffic (all 3 stages), get_flagged_nodes, get_flag_scores, apply_penalties, node_color, node_status
- [ ] `backend/routing_engine.py` — dijkstra, bellman_ford, astar, bfs_path, dfs_path, floyd_warshall, prim_mst, max_flow, topological_sort, compare_all_algorithms
- [ ] `backend/adaptive_router.py` — reset_registry, register_path, sample_path_score, record_outcome, thompson_select, greedy_reroute, dp_optimal_reroute, get_convergence_data, is_path_safe
- [ ] `backend/simulation.py` — run_simulation (full T-step loop)
- [ ] `backend/main.py` — FastAPI app with all 18 endpoints, CORS enabled
- [ ] `frontend/app.py` — Streamlit with sidebar controls and 5 tabs
- [ ] `requirements.txt` — all dependencies pinned to Python 3.11 compatible versions
- [ ] All numpy types cast to Python native types before JSON serialization
- [ ] All functions handle edge cases: no path found, graph not loaded, empty lists

---

## 12. IMPORTANT IMPLEMENTATION NOTES FOR THE IDE

1. **Numpy type serialization:** NetworkX and numpy return numpy.int64, numpy.float64
   types. FastAPI cannot serialize these. Always cast: `int(node)`, `float(value)`,
   `str(label)` before returning from any endpoint.

2. **Module-level state in anomaly_detector.py and adaptive_router.py:**
   These files use module-level dictionaries that persist across API calls.
   This is intentional — it simulates the detector "remembering" history.
   Reset functions are provided to clear this state.

3. **Graph is shared mutable state:**
   The graph G is modified in place (latency penalties, trust scores).
   The simulation resets it via `graph_utils.add_trust_scores(G)` before each
   fresh simulation run.

4. **NetworkX path functions raise exceptions on no-path:**
   Always wrap in try/except NetworkXNoPath, NetworkXUnfeasible, NetworkXError.
   Return `found=False` result dicts rather than crashing.

5. **Betweenness centrality is slow on large graphs:**
   On graphs with 50+ nodes it can take 1-2 seconds. This is acceptable for
   simulation but for the real-time step endpoint, consider using
   `networkx.betweenness_centrality(G, k=10)` (approximate, uses 10 pivot nodes)
   to speed it up.

6. **Streamlit session_state:**
   Store simulation results in `st.session_state["sim_results"]` so they persist
   across tab switches without re-running the simulation.

7. **Plotly graph visualization:**
   Use `networkx.spring_layout(G, seed=42)` to get node positions.
   Convert to Plotly scatter: nodes as markers, edges as line traces.
   Node color comes from GET /graph/nodes endpoint's color field.
