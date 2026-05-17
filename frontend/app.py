import streamlit as st
import httpx
import networkx as nx
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="RouteGuard", layout="wide")

API_BASE = "http://localhost:8000"

def run_simulation(n_nodes, n_edges, seed, source, target, steps, inject_at, spike_nodes, multiplier):
    with httpx.Client(timeout=30.0) as client:
        client.post(f"{API_BASE}/graph/generate", json={"n_nodes": n_nodes, "n_edges": n_edges, "seed": seed})
        client.post(f"{API_BASE}/anomaly/train")
        res = client.post(f"{API_BASE}/anomaly/simulate", json={
            "source": source,
            "target": target,
            "steps": steps,
            "inject_spike_at": inject_at,
            "spike_nodes": spike_nodes,
            "spike_multiplier": float(multiplier)
        })
        if res.status_code == 200:
            st.session_state["sim_results"] = res.json()
            st.session_state["sim_params"] = {"n_nodes": n_nodes, "n_edges": n_edges, "seed": seed, "source": source, "target": target}
        else:
            st.error(f"Error: {res.text}")

# --- Header ---
if "sim_results" in st.session_state:
    sr = st.session_state["sim_results"]
    t_flagged = len(sr["final_flagged"])
    t_paths = len(sr["convergence"])
    header_stats = f"{sr['steps_run']} steps &middot; {t_flagged} flagged &middot; {t_paths} paths"
else:
    header_stats = "0 steps &middot; 0 flagged &middot; 0 paths"

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("### 🔵 **RouteGuard**&nbsp;&nbsp;<span style='color:#777; font-size:14px; font-weight:normal'>adaptive network routing &middot; anomaly-aware path optimization</span>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<p style='text-align: right; color: #777; margin-top:15px; font-size:14px'>{header_stats}</p>", unsafe_allow_html=True)

st.markdown("<hr style='margin-top:0px; margin-bottom:20px'>", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("<p style='font-size:11px; font-weight:bold; color:gray; text-transform:uppercase; margin-bottom:-10px'>Graph</p>", unsafe_allow_html=True)
    n_nodes = st.number_input("Nodes", value=20, min_value=5, max_value=100)
    n_edges = st.number_input("Edges", value=50, min_value=10, max_value=300)
    seed = st.number_input("Seed", value=42)
    st.markdown("<br><p style='font-size:11px; font-weight:bold; color:gray; text-transform:uppercase; margin-bottom:-10px'>Routing</p>", unsafe_allow_html=True)
    source = st.number_input("Source", value=0)
    target = st.number_input("Target", value=15)
    st.markdown("<br><p style='font-size:11px; font-weight:bold; color:gray; text-transform:uppercase; margin-bottom:-10px'>Simulation</p>", unsafe_allow_html=True)
    steps = st.number_input("Steps (T)", value=20)
    inject_at = st.number_input("Inject at step", value=5)
    spike_nodes_str = st.text_input("Spike nodes", value="7,12")
    spike_x = st.number_input("Spike x", value=8.0)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Run Simulation", type="primary", use_container_width=True):
        snodes = [int(x.strip()) for x in spike_nodes_str.split(",") if x.strip()]
        with st.spinner("Running simulation..."):
            run_simulation(n_nodes, n_edges, seed, source, target, steps, inject_at, snodes, spike_x)
            
    st.markdown("---")
    view_step = st.slider("View step", label_visibility="visible", min_value=1, max_value=steps, value=steps) if "sim_results" in st.session_state else 1
    if "sim_results" in st.session_state:
        st.write(f"t = {view_step}")

# --- Tabs ---
t_net, t_route, t_anom, t_reroute, t_algo = st.tabs(["Network Map", "Routing", "Anomaly", "Rerouting Log", "Algorithms"])

def get_step_data(step_idx):
    if "sim_results" not in st.session_state: return None
    tl = st.session_state["sim_results"]["timeline"]
    return tl[min(step_idx - 1, len(tl) - 1)]

step_data = get_step_data(view_step)

with t_net:
    if step_data and "sim_params" in st.session_state:
        c1, c2 = st.columns([3, 1])
        with c1:
            p = st.session_state["sim_params"]
            G_layout = nx.gnm_random_graph(p["n_nodes"], p["n_edges"], seed=p["seed"])
            pos = nx.spring_layout(G_layout, seed=p["seed"])
            
            node_colors = []
            for n in range(p["n_nodes"]):
                status = step_data["node_results"].get(str(n), {}).get("status", "NORMAL")
                if status == "FLAGGED": node_colors.append("#d62728")
                elif status == "WATCHING": node_colors.append("#ff7f0e")
                else: node_colors.append("#2ca02c")
                
            edge_x = []
            edge_y = []
            for u,v in G_layout.edges():
                edge_x.extend([pos[u][0], pos[v][0], None])
                edge_y.extend([pos[u][1], pos[v][1], None])
                
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#ccc'), hoverinfo='none', mode='lines'))
            
            bp = step_data["best_path"]
            if bp:
                bp_x, bp_y = [], []
                for i in range(len(bp)-1):
                    bp_x.extend([pos[bp[i]][0], pos[bp[i+1]][0], None])
                    bp_y.extend([pos[bp[i]][1], pos[bp[i+1]][1], None])
                fig.add_trace(go.Scatter(x=bp_x, y=bp_y, line=dict(width=3, color='#1f77b4'), mode='lines', hoverinfo='none'))
                
            node_x = [pos[k][0] for k in range(p["n_nodes"])]
            node_y = [pos[k][1] for k in range(p["n_nodes"])]
            
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y, mode='markers+text',
                text=[str(n) for n in range(p["n_nodes"])],
                textfont=dict(size=9, color="white"),
                textposition="middle center",
                marker=dict(showscale=False, color=node_colors, size=24, line=dict(width=1, color='white'))
            ))
            fig.update_layout(showlegend=False, margin=dict(b=20, l=20, r=20, t=20), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), plot_bgcolor='#fafafa', height=500)
            
            st.markdown("<div style='border:1px solid #eee; border-radius:5px'>", unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with c2:
            st.markdown("""
            <div style='border: 1px solid #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; background:#fbfbfb'>
                <p style='font-size:11px; font-weight:bold; color:gray; text-transform:uppercase'>Legend</p>
                <div style='display:flex; align-items:center; margin-bottom:10px'><span style='width:14px; height:14px; border-radius:50%; background:#2ca02c; margin-right:10px'></span> Normal</div>
                <div style='display:flex; align-items:center; margin-bottom:10px'><span style='width:14px; height:14px; border-radius:50%; background:#ff7f0e; margin-right:10px'></span> Watching</div>
                <div style='display:flex; align-items:center; margin-bottom:20px'><span style='width:14px; height:14px; border-radius:50%; background:#d62728; margin-right:10px'></span> Flagged</div>
                <div style='display:flex; align-items:center; margin-bottom:10px'><span style='width:30px; height:2px; background:#1f77b4; margin-right:10px'></span> best path</div>
                <div style='display:flex; align-items:center;'><span style='width:30px; height:2px; border-top:2px dotted red; margin-right:10px'></span> penalized edge &times;10</div>
            </div>
            """, unsafe_allow_html=True)
            
            fastest = step_data['algo_results']['fastest_algorithm'] if step_data.get('algo_results') else 'N/A'
            path_str = " &rarr; ".join(map(str, bp)) if bp else "None"
            best_cost = ""
            for ar in step_data.get('algo_results', {}).get('results', []):
                if ar['path'] == bp:
                    best_cost = ar['cost']
                    break
                    
            st.markdown(f"""
            <div style='border: 1px solid #f0f0f0; padding: 15px; border-radius: 5px; background:#fbfbfb'>
                <p style='font-size:11px; font-weight:bold; color:gray; text-transform:uppercase; margin-bottom:15px'>Step {view_step}</p>
                <div style='display:flex; justify-content:space-between; margin-bottom:10px'><span style='color:#555'>best path</span><span style='text-align:right'>{path_str}</span></div>
                <div style='display:flex; justify-content:space-between; margin-bottom:10px'><span style='color:#555'>cost</span><span style='text-align:right'>{best_cost}</span></div>
                <div style='display:flex; justify-content:space-between; margin-bottom:10px'><span style='color:#555'>safe</span><span style='text-align:right; color:{"#2ca02c" if step_data["was_safe"] else "#d62728"}'>{"yes" if step_data["was_safe"] else "no"}</span></div>
                <div style='display:flex; justify-content:space-between; margin-bottom:10px'><span style='color:#555'>flagged</span><span style='text-align:right'>{step_data['flagged_count']}</span></div>
                <div style='display:flex; justify-content:space-between;'><span style='color:#555'>fastest</span><span style='text-align:right'>{fastest}</span></div>
            </div>
            """, unsafe_allow_html=True)

with t_route:
    if step_data:
        algos = step_data["algo_results"]["results"] if step_data.get("algo_results") else []
        if algos:
            df = pd.DataFrame([{
                "Algorithm": a["algorithm"],
                "Path": " → ".join(map(str, a["path"])) if a["path"] else "None",
                "Cost": a["cost"] if a["cost"] != float('inf') else 9999,
                "Hops": a["hops"],
                "Time (ms)": f"{a['time_ms']:.3f}"
            } for a in algos])
            
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<div style='border:1px solid #f0f0f0; border-radius:5px; padding:15px; background:#fff'>", unsafe_allow_html=True)
                st.markdown("<p style='font-size:11px; font-weight:bold; color:gray; text-transform:uppercase'>Cost</p>", unsafe_allow_html=True)
                fig_cost = go.Figure(go.Bar(
                    x=[a["cost"] if a["cost"] != float('inf') else 0 for a in algos],
                    y=[a["algorithm"] for a in algos],
                    orientation='h', marker_color='#1f77b4',
                    text=[f"{a['cost']:.2f}" if a["cost"] != float('inf') else "inf" for a in algos],
                    textposition='outside'
                ))
                fig_cost.update_layout(height=250, margin=dict(b=0, l=0, r=0, t=0), xaxis=dict(visible=False), plot_bgcolor='white')
                fig_cost.update_yaxes(autorange="reversed")
                st.plotly_chart(fig_cost, use_container_width=True, config={'displayModeBar': False})
                st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                st.markdown("<div style='border:1px solid #f0f0f0; border-radius:5px; padding:15px; background:#fff'>", unsafe_allow_html=True)
                st.markdown("<p style='font-size:11px; font-weight:bold; color:gray; text-transform:uppercase'>Time (ms)</p>", unsafe_allow_html=True)
                fig_time = go.Figure(go.Bar(
                    x=[a["time_ms"] for a in algos],
                    y=[a["algorithm"] for a in algos],
                    orientation='h', marker_color='#f0f2f6',
                    text=[f"{a['time_ms']:.2f}" for a in algos],
                    textposition='outside'
                ))
                fig_time.update_layout(height=250, margin=dict(b=0, l=0, r=0, t=0), xaxis=dict(visible=False), plot_bgcolor='white')
                fig_time.update_yaxes(autorange="reversed")
                st.plotly_chart(fig_time, use_container_width=True, config={'displayModeBar': False})
                st.markdown("</div>", unsafe_allow_html=True)

with t_anom:
    if "sim_results" in st.session_state:
        tl = st.session_state["sim_results"]["timeline"]
        st.markdown("<div style='border:1px solid #f0f0f0; padding:15px; border-radius:5px; margin-bottom:20px'>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:11px; font-weight:bold; color:gray; text-transform:uppercase; margin-bottom:10px'>Flagged Count Over Time</p>", unsafe_allow_html=True)
        flagged_counts = [s["flagged_count"] for s in tl]
        colors = ['#e6878b' if c > 0 else '#8bb68b' for c in flagged_counts]
        fig_anom = go.Figure(go.Bar(x=list(range(1, len(tl)+1)), y=flagged_counts, marker_color=colors, opacity=1.0))
        fig_anom.update_layout(height=200, margin=dict(b=20, l=0, r=0, t=0), xaxis=dict(tickfont=dict(color="#bbb"), tickmode='array', tickvals=[1, len(tl)], ticktext=['t=1', f't={len(tl)}']), yaxis=dict(visible=False), plot_bgcolor='white', bargap=0.1)
        st.plotly_chart(fig_anom, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)
        
        if step_data:
            nd = step_data["node_results"]
            df_n = pd.DataFrame([{
                "Node": int(k),
                "Type": v["node_type"],
                "Flag": v["flag_score"],
                "Status": v["status"],
                "Trust": "1.00" 
            } for k,v in nd.items()])
            
            def styling(row):
                return ['color: #2ca02c' if row['Status'] == 'NORMAL' else ('color: #d62728' if row['Status'] == 'FLAGGED' else 'color: #ff7f0e')] * len(row)
            
            st.dataframe(df_n.style.apply(styling, axis=1), use_container_width=True, hide_index=True)

with t_reroute:
    if "sim_results" in st.session_state:
        tl = st.session_state["sim_results"]["timeline"]
        reroutes = [s for s in tl if not s["was_safe"]]
        st.markdown(f"<p style='color:#777; margin-bottom:15px'>{len(reroutes)} reroutes triggered</p>", unsafe_allow_html=True)
        for rr in reroutes:
            bp = rr['best_path']
            st.markdown(f"""
            <div style='border: 1px solid #f0f0f0; padding: 20px; border-radius: 8px; margin-bottom:15px; background:#fff'>
                <div style='display:flex; justify-content:space-between; align-items:baseline; margin-bottom:15px'>
                    <b style='font-size:16px'>Step {rr['step']}</b>
                    <span style='color:#d62728'>flagged: {",".join(map(str, rr['flagged_nodes']))}</span>
                </div>
                <div style='display:flex; justify-content:space-between; color:#555; margin-bottom:5px'>
                    <span>original best</span>
                    <span>{"&rarr;".join(map(str, bp))} cost={rr['algo_results']['results'][0]['cost']}</span>
                </div>
                <div style='display:flex; justify-content:space-between; color:#555; margin-bottom:5px'>
                    <span>greedy</span>
                    <span>{("&rarr;".join(map(str, rr['greedy_reroute']['path'])) + " cost=" + str(rr['greedy_reroute']['cost'])) if rr['greedy_reroute'] and rr['greedy_reroute']['found'] else 'no path'}</span>
                </div>
                <div style='display:flex; justify-content:space-between; color:#555'>
                    <span>dp (bellman-ford)</span>
                    <span>{("&rarr;".join(map(str, rr['dp_reroute']['path'])) + " cost=" + str(rr['dp_reroute']['cost'])) if rr['dp_reroute'] and rr['dp_reroute']['found'] else 'no path'}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
with t_algo:
    st.markdown("""
    | Algorithm | Category | Complexity | Role |
    |---|---|---|---|
    | Dijkstra | Graph | O((V+E) log V) | Primary routing |
    | Bellman-Ford | Graph | O(VE) | Penalty-aware routing |
    | A* | Graph | O(E log V) | Heuristic routing |
    | BFS | Traversal | O(V+E) | Hop-minimal path |
    | DFS | Traversal | O(V+E) | Path auditing |
    | Floyd-Warshall | DP | O(V³) | DP rerouting |
    | Prim's MST | Greedy | O(E log V) | Network backbone |
    | Max-Flow | Flow | O(VE²) | Bottleneck detection |
    | Betweenness Centrality | Graph | O(VE) | Anomaly detection |
    | Topological Sort | DAG | O(V+E) | Packet ordering |
    | Isolation Forest | ML | O(n log n) | Multivariate anomaly |
    | Thompson Sampling | Probabilistic | O(1) | Adaptive path selection |
    """)
    if "sim_results" in st.session_state:
        st.markdown("<br><div style='border:1px solid #f0f0f0; padding:20px; border-radius:8px; background:#fff'>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:11px; font-weight:bold; color:gray; text-transform:uppercase; margin-bottom:15px'>Thompson Sampling Convergence (Top 2 paths by confidence)</p>", unsafe_allow_html=True)
        conv = st.session_state["sim_results"]["convergence"]
        for c in conv[:2]:
            width_pct = int(c['mean'] * 100)
            color = "#d62728" if c['mean'] < 0.6 else "#2ca02c"
            disp_path = c['path_key'].replace("[", "").replace("]", "").replace(", ", "→")
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:15px'>
                <div style='width:25%; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-family:monospace; font-size:13px; color:#555'>{disp_path}</div>
                <div style='width:60%; background:#f5f5f5; height:18px; border-radius:4px; margin-left:10px; margin-right:10px'><div style='width:{width_pct}%; background:{color}; height:100%; border-radius:4px'></div></div>
                <div style='width:15%; text-align:right; font-family:monospace; font-size:13px; color:#444'>&mu;={c['mean']:.2f} &nbsp;&nbsp;&nbsp; n={c['confidence']}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
