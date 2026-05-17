import numpy as np

import anomaly_detector as ad
import routing_engine as re
import adaptive_router as ar

def run_simulation(G, source, target, T, inject_at, inject_nodes, spike_multiplier):
    timeline = []
    
    for t in range(1, T + 1):
        # Step 1 — Generate traffic volumes
        volumes = {}
        for node in G.nodes():
            vol = max(10.0, float(np.random.normal(100, 25)))
            if t == inject_at and inject_nodes and node in inject_nodes:
                vol *= spike_multiplier
            volumes[int(node)] = float(vol)
            
        # Step 2 — Run anomaly detection
        node_results = ad.update_traffic(G, volumes)
        
        # Step 3 — Apply penalties
        ad.apply_penalties(G)
        
        # Step 4 — Get current flagged nodes
        flagged = ad.get_flagged_nodes()
        
        # Step 5 — Run routing algorithms
        algo_results = re.compare_all_algorithms(G, source, target)
        
        # Step 6 — Thompson Sampling selection
        candidate_paths = [r["path"] for r in algo_results["results"] if r["found"] and r["path"]]
        best_path = ar.thompson_select(candidate_paths)
        
        # Step 7 — Record outcome
        if best_path:
            was_safe = ar.is_path_safe(best_path, flagged)
            ar.record_outcome(best_path, was_safe)
        else:
            was_safe = True
            
        # Step 8 — Reroute if needed
        if best_path and not was_safe:
            greedy_result = ar.greedy_reroute(G, source, target, flagged)
            dp_result = ar.dp_optimal_reroute(G, source, target, flagged)
        else:
            greedy_result = None
            dp_result = None
            
        # Step 9 — MST every 5 steps
        mst_result = re.prim_mst(G) if t % 5 == 0 else None
        
        # Step 10 — Max-flow every 10 steps
        flow_result = re.max_flow(G, source, target) if t % 10 == 0 else None
        
        # Step 11 — Topological sort every 5 steps
        topo_result = re.topological_sort(G) if t % 5 == 0 else None
        
        # Store per-timestep log dict
        timeline.append({
            "step": t,
            "spike_injected": (t == inject_at),
            "volumes": volumes,
            "flagged_nodes": flagged,
            "watching_nodes": [n for n, r in node_results.items() if r["status"] == "WATCHING"],
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
        })
        
    return {
        "steps_run": T,
        "source": source,
        "target": target,
        "timeline": timeline,
        "final_flagged": ad.get_flagged_nodes(),
        "final_scores": ad.get_flag_scores(),
        "convergence": ar.get_convergence_data(),
        "final_node_colors": {int(n): ad.node_color(n) for n in G.nodes()},
    }
