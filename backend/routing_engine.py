import time
import collections
import networkx as nx

def _format_result(algo_name, path, cost, hops, time_ms, found):
    return {
        "algorithm": algo_name,
        "path": [int(n) for n in path],
        "cost": float(cost) if cost != float('inf') else float('inf'),
        "hops": hops,
        "time_ms": float(time_ms),
        "found": found
    }

def dijkstra(G, source, target):
    start = time.perf_counter()
    try:
        path = nx.dijkstra_path(G, source, target, weight='latency')
        cost = nx.dijkstra_path_length(G, source, target, weight='latency')
        found = True
    except nx.NetworkXNoPath:
        path = []
        cost = float('inf')
        found = False
    except nx.NodeNotFound:
        path = []
        cost = float('inf')
        found = False
    
    time_ms = (time.perf_counter() - start) * 1000
    hops = len(path) - 1 if path else 0
    return _format_result("Dijkstra", path, cost, hops, time_ms, found)

def bellman_ford(G, source, target):
    start = time.perf_counter()
    try:
        path = nx.bellman_ford_path(G, source, target, weight='latency')
        cost = nx.bellman_ford_path_length(G, source, target, weight='latency')
        found = True
    except nx.NetworkXNoPath:
        path = []
        cost = float('inf')
        found = False
    except nx.NetworkXUnbounded:
        path = []
        cost = float('inf')
        found = False
    except nx.NodeNotFound:
        path = []
        cost = float('inf')
        found = False
        
    time_ms = (time.perf_counter() - start) * 1000
    hops = len(path) - 1 if path else 0
    return _format_result("Bellman-Ford", path, cost, hops, time_ms, found)

def astar(G, source, target):
    start = time.perf_counter()
    try:
        heuristic = lambda u, v: abs(u - v) * 0.5
        path = nx.astar_path(G, source, target, weight='latency', heuristic=heuristic)
        if len(path) > 1:
            cost = sum(G.edges[path[i], path[i+1]]['latency'] for i in range(len(path)-1))
        else:
            cost = 0.0
        found = True
    except nx.NetworkXNoPath:
        path = []
        cost = float('inf')
        found = False
    except nx.NodeNotFound:
        path = []
        cost = float('inf')
        found = False
        
    time_ms = (time.perf_counter() - start) * 1000
    hops = len(path) - 1 if path else 0
    return _format_result("A*", path, cost, hops, time_ms, found)

def bfs_path(G, source, target):
    start = time.perf_counter()
    if source not in G or target not in G:
        return _format_result("BFS", [], float('inf'), 0, (time.perf_counter() - start) * 1000, False)
        
    queue = collections.deque([[source]])
    visited = {source}
    path = []
    found = False
    while queue:
        curr_path = queue.popleft()
        node = curr_path[-1]
        
        if node == target:
            path = curr_path
            found = True
            break
            
        for nbr in G.successors(node):
            if nbr not in visited:
                visited.add(nbr)
                queue.append(curr_path + [nbr])
    
    cost = sum(G.edges[path[i], path[i+1]]['latency'] for i in range(len(path)-1)) if found else float('inf')
    time_ms = (time.perf_counter() - start) * 1000
    hops = len(path) - 1 if path else 0
    return _format_result("BFS", path, cost, hops, time_ms, found)

def dfs_path(G, source, target):
    start = time.perf_counter()
    if source not in G or target not in G:
        return _format_result("DFS", [], float('inf'), 0, (time.perf_counter() - start) * 1000, False)

    stack = [[source]]
    visited = {source}
    path = []
    found = False
    while stack:
        curr_path = stack.pop()
        node = curr_path[-1]
        
        if node == target:
            path = curr_path
            found = True
            break
            
        for nbr in G.successors(node):
            if nbr not in visited:
                visited.add(nbr)
                stack.append(curr_path + [nbr])
                
    cost = sum(G.edges[path[i], path[i+1]]['latency'] for i in range(len(path)-1)) if found else float('inf')
    time_ms = (time.perf_counter() - start) * 1000
    hops = len(path) - 1 if path else 0
    return _format_result("DFS", path, cost, hops, time_ms, found)

def floyd_warshall(G):
    pred, dist = nx.floyd_warshall_predecessor_and_distance(G, weight='latency')
    return {
        "distance_matrix": {int(k): {int(k2): float(v2) for k2, v2 in v.items()} for k, v in dist.items()},
        "predecessor": {int(k): {int(k2): int(v2) for k2, v2 in v.items()} for k, v in pred.items()}
    }

def prim_mst(G):
    G_undirected = G.to_undirected()
    try:
        mst = nx.minimum_spanning_tree(G_undirected, weight='latency')
        edges = [(int(u), int(v), float(d['latency'])) for u, v, d in mst.edges(data=True)]
        total_cost = sum(d['latency'] for u, v, d in mst.edges(data=True))
        node_count = mst.number_of_nodes()
    except Exception:
        edges = []
        total_cost = 0.0
        node_count = 0
        
    return {
        "edges": edges,
        "total_cost": float(total_cost),
        "node_count": int(node_count)
    }

def max_flow(G, source, sink):
    if source not in G or sink not in G:
        return {"flow_value": 0.0, "cut_value": 0.0, "reachable": [], "non_reachable": []}
    
    try:
        flow_value = nx.maximum_flow_value(G, source, sink, capacity='bandwidth')
        cut_value, (reachable, non_reachable) = nx.minimum_cut(G, source, sink, capacity='bandwidth')
        return {
            "flow_value": float(flow_value),
            "cut_value": float(cut_value),
            "reachable": [int(n) for n in reachable],
            "non_reachable": [int(n) for n in non_reachable]
        }
    except Exception:
        return {
            "flow_value": 0.0,
            "cut_value": 0.0,
            "reachable": [],
            "non_reachable": []
        }

def topological_sort(G):
    try:
        order = list(nx.topological_sort(G))
        return {
            "success": True,
            "cycle_detected": False,
            "order": [int(n) for n in order]
        }
    except nx.NetworkXUnfeasible:
        return {
            "success": False,
            "cycle_detected": True,
            "order": []
        }

def compare_all_algorithms(G, source, target):
    results = [
        dijkstra(G, source, target),
        bellman_ford(G, source, target),
        astar(G, source, target),
        bfs_path(G, source, target),
        dfs_path(G, source, target)
    ]
    
    found_results = [r for r in results if r['found']]
    if found_results:
        fastest = min(found_results, key=lambda x: x['time_ms'])
        fastest_name = fastest['algorithm']
    else:
        fastest_name = "None"
        
    return {
        "results": results,
        "fastest_algorithm": fastest_name
    }
