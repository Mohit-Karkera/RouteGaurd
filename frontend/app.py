import streamlit as st
import httpx
import networkx as nx
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="RouteGuard", layout="wide")

API_BASE = "http://localhost:8000"

# ── Exact colors from screenshot ──────────────────────────────────────────────
C_NORMAL   = "#3d9970"   # green
C_WATCHING = "#ff851b"   # orange
C_FLAGGED  = "#e74c3c"   # red
C_PATH     = "#2e86de"   # bright blue — best path line
C_EDGE     = "#d0d0d0"   # light grey edges
C_BG       = "#ffffff"   # pure white graph background
C_PANEL    = "#fafafa"   # panel backgrounds

def node_color(status):
    if status == "FLAGGED":  return C_FLAGGED
    if status == "WATCHING": return C_WATCHING
    return C_NORMAL

# ── API helpers ───────────────────────────────────────────────────────────────
def run_simulation(n_nodes, n_edges, seed, source, target, steps, inject_at, spike_nodes, multiplier):
    with httpx.Client(timeout=60.0) as client:
        r = client.post(f"{API_BASE}/graph/generate",
                        json={"n_nodes": n_nodes, "n_edges": n_edges, "seed": seed})
        if r.status_code != 200:
            st.error(f"Graph generation failed: {r.text}"); return
        client.post(f"{API_BASE}/anomaly/train")
        res = client.post(f"{API_BASE}/anomaly/simulate", json={
            "source": source, "target": target,
            "steps": steps, "inject_spike_at": inject_at,
            "spike_nodes": spike_nodes, "spike_multiplier": float(multiplier)
        })
        if res.status_code == 200:
            st.session_state["sim_results"] = res.json()
            st.session_state["sim_params"]  = {
                "n_nodes": n_nodes, "n_edges": n_edges,
                "seed": seed, "source": source, "target": target
            }
        else:
            st.error(f"Simulation failed: {res.text}")

# ── Header ────────────────────────────────────────────────────────────────────
if "sim_results" in st.session_state:
    sr = st.session_state["sim_results"]
    header_stats = (f"{sr['steps_run']} steps &middot; "
                    f"{len(sr['final_flagged'])} flagged &middot; "
                    f"{len(sr['convergence'])} paths")
else:
    header_stats = "0 steps &middot; 0 flagged &middot; 0 paths"

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(
        "### 🔵 **RouteGuard**&nbsp;&nbsp;"
        "<span style='color:#777; font-size:14px; font-weight:normal'>"
        "adaptive network routing &middot; anomaly-aware path optimization</span>",
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        f"<p style='text-align:right; color:#777; margin-top:15px; font-size:14px'>{header_stats}</p>",
        unsafe_allow_html=True
    )
st.markdown("<hr style='margin-top:0; margin-bottom:20px'>", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    def section(label):
        st.markdown(
            f"<p style='font-size:11px; font-weight:bold; color:gray; "
            f"text-transform:uppercase; margin-bottom:-10px'>{label}</p>",
            unsafe_allow_html=True
        )

    section("Graph")
    n_nodes = st.number_input("Nodes",  value=20, min_value=5,  max_value=100)
    n_edges = st.number_input("Edges",  value=50, min_value=10, max_value=300)
    seed    = st.number_input("Seed",   value=42)

    st.markdown("<br>", unsafe_allow_html=True)
    section("Routing")
    source  = st.number_input("Source", value=0)
    target  = st.number_input("Target", value=15)

    st.markdown("<br>", unsafe_allow_html=True)
    section("Simulation")
    steps          = st.number_input("Steps (T)",       value=20)
    inject_at      = st.number_input("Inject at step",  value=5)
    spike_nodes_str = st.text_input("Spike nodes",      value="7,12")
    spike_x        = st.number_input("Spike x",         value=8.0)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Run Simulation", type="primary", use_container_width=True):
        snodes = [int(x.strip()) for x in spike_nodes_str.split(",") if x.strip().isdigit()]
        with st.spinner("Running simulation..."):
            run_simulation(n_nodes, n_edges, seed, source, target,
                           steps, inject_at, snodes, spike_x)

    st.markdown("---")
    if "sim_results" in st.session_state:
        view_step = st.slider("View step", min_value=1, max_value=int(steps), value=int(steps))
        st.write(f"t = {view_step}")
    else:
        view_step = 1

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_step_data(step_idx):
    if "sim_results" not in st.session_state: return None
    tl = st.session_state["sim_results"]["timeline"]
    return tl[min(step_idx - 1, len(tl) - 1)]

step_data = get_step_data(view_step)

def build_graph_fig(p, step_data):
    """Build Plotly network map matching the screenshot exactly."""
    G_layout = nx.gnm_random_graph(p["n_nodes"], p["n_edges"], seed=p["seed"])
    pos = nx.spring_layout(G_layout, seed=p["seed"])

    src, tgt = p["source"], p["target"]

    # Node statuses
    node_statuses = {}
    if step_data:
        for k, v in step_data["node_results"].items():
            node_statuses[int(k)] = v.get("status", "NORMAL")

    # ── Traces ────────────────────────────────────────────────────────────────
    fig = go.Figure()

    # 1. All edges — light grey
    edge_x, edge_y = [], []
    for u, v in G_layout.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.7, color=C_EDGE),
        hoverinfo="none", mode="lines"
    ))

    # 2. Best path — bright blue, width 3
    if step_data:
        bp = step_data.get("best_path") or []
        if len(bp) >= 2:
            bp_x, bp_y = [], []
            for i in range(len(bp) - 1):
                bp_x += [pos[bp[i]][0], pos[bp[i+1]][0], None]
                bp_y += [pos[bp[i]][1], pos[bp[i+1]][1], None]
            fig.add_trace(go.Scatter(
                x=bp_x, y=bp_y,
                line=dict(width=3, color=C_PATH),
                mode="lines", hoverinfo="none"
            ))

    # 3. Regular nodes (not source/target)
    groups = {
        "NORMAL":   {"color": C_NORMAL,   "nodes": []},
        "WATCHING": {"color": C_WATCHING, "nodes": []},
        "FLAGGED":  {"color": C_FLAGGED,  "nodes": []},
    }
    for n in G_layout.nodes():
        if n in (src, tgt): continue
        status = node_statuses.get(n, "NORMAL")
        groups[status]["nodes"].append(n)

    for status, g in groups.items():
        if not g["nodes"]: continue
        nx_list = g["nodes"]
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in nx_list],
            y=[pos[n][1] for n in nx_list],
            mode="markers+text",
            text=[str(n) for n in nx_list],
            textfont=dict(size=9, color="white"),
            textposition="middle center",
            marker=dict(
                color=g["color"], size=26,
                line=dict(width=1.5, color="white")
            ),
            hovertemplate=[
                f"Node {n}<br>Status: {status}<br>Flag score: "
                f"{step_data['node_results'].get(str(n),{}).get('flag_score',0) if step_data else 0}"
                for n in nx_list
            ],
            name=status
        ))

    # 4. Source node — white ring border, slightly larger
    if src in pos:
        s_status = node_statuses.get(src, "NORMAL")
        fig.add_trace(go.Scatter(
            x=[pos[src][0]], y=[pos[src][1]],
            mode="markers+text",
            text=[str(src)],
            textfont=dict(size=9, color="white"),
            textposition="middle center",
            marker=dict(
                color=node_color(s_status), size=28,
                line=dict(width=3, color="white")
            ),
            hovertemplate=f"Node {src} (SOURCE)<br>Status: {s_status}",
            name="Source"
        ))

    # 5. Target node — white ring border, slightly larger
    if tgt in pos and tgt != src:
        t_status = node_statuses.get(tgt, "NORMAL")
        fig.add_trace(go.Scatter(
            x=[pos[tgt][0]], y=[pos[tgt][1]],
            mode="markers+text",
            text=[str(tgt)],
            textfont=dict(size=9, color="white"),
            textposition="middle center",
            marker=dict(
                color=node_color(t_status), size=28,
                line=dict(width=3, color="white")
            ),
            hovertemplate=f"Node {tgt} (TARGET)<br>Status: {t_status}",
            name="Target"
        ))

    fig.update_layout(
        showlegend=False,
        margin=dict(b=20, l=20, r=20, t=20),
        plot_bgcolor=C_BG,
        paper_bgcolor=C_BG,
        height=500,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
    )
    return fig

# ── Tabs ──────────────────────────────────────────────────────────────────────
t_net, t_route, t_anom, t_reroute, t_algo = st.tabs(
    ["Network Map", "Routing", "Anomaly", "Rerouting Log", "Algorithms"]
)

# ────────────────────────────────────────────────────────── Tab 1: Network Map
with t_net:
    if step_data and "sim_params" in st.session_state:
        p = st.session_state["sim_params"]
        c1, c2 = st.columns([3, 1])

        with c1:
            fig = build_graph_fig(p, step_data)
            st.markdown(
                "<div style='border:1px solid #eee; border-radius:8px; overflow:hidden'>",
                unsafe_allow_html=True
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            # Legend panel
            st.markdown(f"""
            <div style='border:1px solid #f0f0f0; padding:16px; border-radius:8px;
                        margin-bottom:16px; background:{C_PANEL}'>
                <p style='font-size:11px; font-weight:bold; color:gray;
                          text-transform:uppercase; margin-bottom:14px; letter-spacing:.05em'>Legend</p>
                <div style='display:flex; align-items:center; margin-bottom:10px'>
                    <span style='width:13px; height:13px; border-radius:50%;
                                 background:{C_NORMAL}; margin-right:10px; flex-shrink:0'></span>
                    <span style='font-size:13px; color:#444'>Normal</span>
                </div>
                <div style='display:flex; align-items:center; margin-bottom:10px'>
                    <span style='width:13px; height:13px; border-radius:50%;
                                 background:{C_WATCHING}; margin-right:10px; flex-shrink:0'></span>
                    <span style='font-size:13px; color:#444'>Watching</span>
                </div>
                <div style='display:flex; align-items:center; margin-bottom:18px'>
                    <span style='width:13px; height:13px; border-radius:50%;
                                 background:{C_FLAGGED}; margin-right:10px; flex-shrink:0'></span>
                    <span style='font-size:13px; color:#444'>Flagged</span>
                </div>
                <div style='display:flex; align-items:center; margin-bottom:10px'>
                    <span style='display:inline-block; width:28px; height:2px;
                                 background:{C_PATH}; margin-right:10px; flex-shrink:0'></span>
                    <span style='font-size:13px; color:#444'>best path</span>
                </div>
                <div style='display:flex; align-items:center'>
                    <span style='display:inline-block; width:28px; height:0;
                                 border-top:2px dashed #e74c3c; margin-right:10px; flex-shrink:0'></span>
                    <span style='font-size:13px; color:#444'>penalized edge &times;10</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Step info panel
            bp = step_data.get("best_path") or []
            path_str = " &rarr; ".join(map(str, bp)) if bp else "None"
            best_cost = "—"
            for ar in step_data.get("algo_results", {}).get("results", []):
                if ar.get("path") == bp and ar.get("cost") not in (None, float("inf")):
                    best_cost = ar["cost"]
                    break
            fastest = step_data.get("algo_results", {}).get("fastest_algorithm", "N/A")
            safe_val = step_data.get("was_safe", True)
            safe_color = C_NORMAL if safe_val else C_FLAGGED
            safe_text  = "yes" if safe_val else "no"

            def info_row(label, value, value_color="#222"):
                return (f"<div style='display:flex; justify-content:space-between; "
                        f"align-items:baseline; margin-bottom:10px'>"
                        f"<span style='color:#888; font-size:13px'>{label}</span>"
                        f"<span style='color:{value_color}; font-size:13px; font-weight:500'>"
                        f"{value}</span></div>")

            st.markdown(f"""
            <div style='border:1px solid #f0f0f0; padding:16px; border-radius:8px;
                        background:{C_PANEL}'>
                <p style='font-size:11px; font-weight:bold; color:gray;
                          text-transform:uppercase; margin-bottom:14px; letter-spacing:.05em'>
                    Step {view_step}</p>
                {info_row("best path", path_str)}
                {info_row("cost",      best_cost)}
                {info_row("safe",      safe_text, safe_color)}
                {info_row("flagged",   step_data.get("flagged_count", 0))}
                {info_row("fastest",   fastest)}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Run a simulation to see the network map.")

# ─────────────────────────────────────────────────────────── Tab 2: Routing
with t_route:
    if step_data:
        algos = (step_data.get("algo_results") or {}).get("results", [])
        if algos:
            df = pd.DataFrame([{
                "Algorithm" : a["algorithm"],
                "Path"      : " → ".join(map(str, a["path"])) if a.get("path") else "None",
                "Cost"      : round(a["cost"], 2) if a.get("cost") not in (None, float("inf")) else "∞",
                "Hops"      : a.get("hops", 0),
                "Time (ms)" : f"{a['time_ms']:.3f}" if a.get("time_ms") is not None else "—",
            } for a in algos])

            st.dataframe(df, use_container_width=True, hide_index=True)
            st.markdown("<br>", unsafe_allow_html=True)

            c1, c2 = st.columns(2)

            def bar_chart(algos, value_key, title, bar_color, label_fmt="{:.2f}"):
                values = []
                for a in algos:
                    v = a.get(value_key, 0)
                    values.append(0 if v in (None, float("inf")) else v)
                labels = [label_fmt.format(v) for v in values]
                fig = go.Figure(go.Bar(
                    x=values, y=[a["algorithm"] for a in algos],
                    orientation="h",
                    marker=dict(color=bar_color, line=dict(width=0)),
                    text=labels, textposition="outside",
                    textfont=dict(size=11, color="#555")
                ))
                fig.update_layout(
                    height=240,
                    margin=dict(b=0, l=0, r=60, t=0),
                    xaxis=dict(visible=False),
                    yaxis=dict(autorange="reversed", tickfont=dict(size=12, color="#555")),
                    plot_bgcolor="white", paper_bgcolor="white"
                )
                return fig

            with c1:
                st.markdown(
                    f"<div style='border:1px solid #f0f0f0; border-radius:8px; "
                    f"padding:16px; background:{C_PANEL}'>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    "<p style='font-size:11px; font-weight:bold; color:gray; "
                    "text-transform:uppercase; margin-bottom:8px'>Cost (ms latency)</p>",
                    unsafe_allow_html=True
                )
                st.plotly_chart(
                    bar_chart(algos, "cost", "Cost", C_PATH),
                    use_container_width=True, config={"displayModeBar": False}
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with c2:
                st.markdown(
                    f"<div style='border:1px solid #f0f0f0; border-radius:8px; "
                    f"padding:16px; background:{C_PANEL}'>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    "<p style='font-size:11px; font-weight:bold; color:gray; "
                    "text-transform:uppercase; margin-bottom:8px'>Execution Time (ms)</p>",
                    unsafe_allow_html=True
                )
                st.plotly_chart(
                    bar_chart(algos, "time_ms", "Time", "#e8ecf0", "{:.3f}"),
                    use_container_width=True, config={"displayModeBar": False}
                )
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Run a simulation to see routing analysis.")

# ─────────────────────────────────────────────────────────── Tab 3: Anomaly
with t_anom:
    if "sim_results" in st.session_state:
        tl  = st.session_state["sim_results"]["timeline"]
        fc  = [s["flagged_count"] for s in tl]
        bar_colors = [C_FLAGGED if c > 0 else "#b8dfb8" for c in fc]

        st.markdown(
            f"<div style='border:1px solid #f0f0f0; padding:16px; border-radius:8px; "
            f"margin-bottom:20px; background:{C_PANEL}'>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='font-size:11px; font-weight:bold; color:gray; "
            "text-transform:uppercase; margin-bottom:8px'>Flagged Count Over Time</p>",
            unsafe_allow_html=True
        )
        fig_anom = go.Figure(go.Bar(
            x=list(range(1, len(tl) + 1)), y=fc,
            marker_color=bar_colors
        ))
        fig_anom.update_layout(
            height=200,
            margin=dict(b=20, l=0, r=0, t=0),
            xaxis=dict(
                tickfont=dict(color="#bbb"),
                tickmode="array",
                tickvals=[1, len(tl)],
                ticktext=["t=1", f"t={len(tl)}"]
            ),
            yaxis=dict(visible=False),
            plot_bgcolor="white", paper_bgcolor="white",
            bargap=0.1
        )
        st.plotly_chart(fig_anom, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

        if step_data:
            nd = step_data["node_results"]
            rows = []
            for k, v in nd.items():
                rows.append({
                    "Node"      : int(k),
                    "Type"      : v["node_type"],
                    "Flag Score": v["flag_score"],
                    "Z-Score"   : round(v.get("z_score", 0), 3),
                    "Centrality": round(v.get("centrality", 0), 4),
                    "Status"    : v["status"],
                })
            df_n = pd.DataFrame(rows).sort_values("Flag Score", ascending=False)

            def row_style(row):
                if row["Status"] == "FLAGGED":  c = C_FLAGGED
                elif row["Status"] == "WATCHING": c = C_WATCHING
                else:                             c = C_NORMAL
                return [f"color:{c}"] * len(row)

            st.dataframe(
                df_n.style.apply(row_style, axis=1),
                use_container_width=True, hide_index=True
            )
    else:
        st.info("Run a simulation to see anomaly report.")

# ──────────────────────────────────────────────────────── Tab 4: Rerouting Log
with t_reroute:
    if "sim_results" in st.session_state:
        tl       = st.session_state["sim_results"]["timeline"]
        reroutes = [s for s in tl if not s.get("was_safe", True)]

        st.markdown(
            f"<p style='color:#777; margin-bottom:15px'>{len(reroutes)} reroutes triggered</p>",
            unsafe_allow_html=True
        )

        if not reroutes:
            st.success("No reroutes — all paths were safe throughout the simulation.")

        for rr in reroutes:
            bp   = rr.get("best_path") or []
            orig_cost = "—"
            for ar in rr.get("algo_results", {}).get("results", []):
                if ar.get("cost") not in (None, float("inf")):
                    orig_cost = round(ar["cost"], 2)
                    break

            gr   = rr.get("greedy_reroute") or {}
            dp   = rr.get("dp_reroute")     or {}

            gr_str = (" &rarr; ".join(map(str, gr["path"])) + f" cost={gr['cost']}"
                      if gr.get("found") else "no path found")
            dp_str = (" &rarr; ".join(map(str, dp["path"])) + f" cost={dp['cost']}"
                      if dp.get("found") else "no path found")

            def rr_row(label, value):
                return (f"<div style='display:flex; justify-content:space-between; "
                        f"color:#555; font-size:13px; margin-bottom:8px'>"
                        f"<span style='color:#999'>{label}</span>"
                        f"<span style='font-family:monospace'>{value}</span></div>")

            st.markdown(f"""
            <div style='border:1px solid #f0f0f0; padding:20px; border-radius:8px;
                        margin-bottom:14px; background:{C_PANEL}'>
                <div style='display:flex; justify-content:space-between;
                            align-items:baseline; margin-bottom:14px'>
                    <b style='font-size:15px'>Step {rr['step']}</b>
                    <span style='color:{C_FLAGGED}; font-size:13px'>
                        flagged: {", ".join(map(str, rr.get("flagged_nodes", [])))}
                    </span>
                </div>
                {rr_row("original best", " &rarr; ".join(map(str, bp)) + f" cost={orig_cost}")}
                {rr_row("greedy", gr_str)}
                {rr_row("dp (floyd-warshall)", dp_str)}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Run a simulation to see the rerouting log.")

# ──────────────────────────────────────────────────────── Tab 5: Algorithms
with t_algo:
    st.markdown("""
| Algorithm | Category | Complexity | Role in RouteGuard |
|---|---|---|---|
| Dijkstra | Graph | O((V+E) log V) | Primary routing baseline |
| Bellman-Ford | Graph | O(VE) | Penalty-aware routing |
| A* | Graph | O(E log V) | Heuristic routing |
| BFS | Traversal | O(V+E) | Hop-minimal path + alert spread |
| DFS | Traversal | O(V+E) | Path auditing |
| Floyd-Warshall | DP | O(V³) | DP optimal rerouting |
| Prim's MST | Greedy | O(E log V) | Network backbone |
| Max-Flow | Flow | O(VE²) | Bottleneck detection |
| Betweenness Centrality | Graph | O(VE) | Anomaly Stage 2 |
| Topological Sort | DAG | O(V+E) | Packet ordering / cycle detection |
| Isolation Forest | ML | O(n log n) | Anomaly Stage 3 |
| Thompson Sampling | Probabilistic | O(1) | Adaptive path selection |
""")

    if "sim_results" in st.session_state:
        conv = st.session_state["sim_results"].get("convergence", [])
        if conv:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='border:1px solid #f0f0f0; padding:20px; border-radius:8px; "
                f"background:{C_PANEL}'>",
                unsafe_allow_html=True
            )
            st.markdown(
                "<p style='font-size:11px; font-weight:bold; color:gray; "
                "text-transform:uppercase; margin-bottom:16px; letter-spacing:.05em'>"
                "Thompson Sampling Convergence — top paths by confidence</p>",
                unsafe_allow_html=True
            )
            for c in conv[:4]:
                mean        = c.get("mean", 0)
                confidence  = c.get("confidence", 0)
                width_pct   = int(mean * 100)
                bar_color   = C_NORMAL if mean >= 0.6 else (C_WATCHING if mean >= 0.4 else C_FLAGGED)
                path_key    = c.get("path_key", "")
                disp_path   = path_key.replace("[", "").replace("]", "").replace(", ", " → ")

                st.markdown(f"""
                <div style='display:flex; align-items:center; margin-bottom:14px'>
                    <div style='width:24%; white-space:nowrap; overflow:hidden;
                                text-overflow:ellipsis; font-family:monospace;
                                font-size:12px; color:#666'>{disp_path}</div>
                    <div style='flex:1; background:#f0f0f0; height:16px;
                                border-radius:4px; margin:0 12px; overflow:hidden'>
                        <div style='width:{width_pct}%; background:{bar_color};
                                    height:100%; border-radius:4px;
                                    transition: width .3s ease'></div>
                    </div>
                    <div style='width:18%; text-align:right; font-family:monospace;
                                font-size:12px; color:#555'>
                        &mu;={mean:.2f} &nbsp; n={confidence}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)