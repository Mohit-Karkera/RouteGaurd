import sys
import json
sys.path.insert(0, 'backend')
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)
endpoints_tested = 0
failed = []

def run_test(name, req_fn):
    global endpoints_tested, failed
    print(f"Testing {name}...", end=" ")
    try:
        res = req_fn()
        if res.status_code == 200:
            print("OK")
            endpoints_tested += 1
            return res.json()
        else:
            print(f"FAILED (Status {res.status_code}): {res.text}")
            failed.append(name)
            return None
    except Exception as e:
        print(f"ERROR: {e}")
        failed.append(name)
        return None

# 1. Root
run_test("GET /", lambda: client.get("/"))

# 2. Generate Graph
generate_data = run_test("POST /graph/generate", lambda: client.post("/graph/generate", json={"n_nodes": 20, "n_edges": 60, "seed": 42}))
nodes = generate_data["nodes"]
source = nodes[0]
target = nodes[-1]

# 3. Graph Summary
run_test("GET /graph/summary", lambda: client.get("/graph/summary"))

# 4. Graph Nodes
run_test("GET /graph/nodes", lambda: client.get("/graph/nodes"))

# 5. Anomaly Train
run_test("POST /anomaly/train", lambda: client.post("/anomaly/train"))

# 6. Anomaly Step
run_test("POST /anomaly/step", lambda: client.post("/anomaly/step", json={"inject_spike": [source], "spike_multiplier": 5.0}))

# 7. Anomaly Flagged
run_test("GET /anomaly/flagged", lambda: client.get("/anomaly/flagged"))

# 8. Anomaly Scores
run_test("GET /anomaly/scores", lambda: client.get("/anomaly/scores"))

# 9. Anomaly Node Detail
run_test(f"GET /anomaly/node/{source}", lambda: client.get(f"/anomaly/node/{source}"))

# 10. Anomaly History
run_test("GET /anomaly/history", lambda: client.get("/anomaly/history"))

# 11. Anomaly Reset
run_test("POST /anomaly/reset", lambda: client.post("/anomaly/reset"))

# Re-train after reset just in case
client.post("/anomaly/train")

# 12. Routing Compare
run_test("GET /routing/compare", lambda: client.get(f"/routing/compare?source={source}&target={target}"))

# 13. Routing MST
run_test("GET /routing/mst", lambda: client.get("/routing/mst"))

# 14. Routing MaxFlow
run_test("GET /routing/maxflow", lambda: client.get(f"/routing/maxflow?source={source}&sink={target}"))

# 15. Routing Topological
run_test("GET /routing/topological", lambda: client.get("/routing/topological"))

# 16. Anomaly Simulate
run_test("POST /anomaly/simulate", lambda: client.post("/anomaly/simulate", json={"source": source, "target": target, "steps": 5, "inject_spike_at": 2, "spike_nodes": [source], "spike_multiplier": 8.0}))

# 17. Simulation Results
run_test("GET /simulation/results", lambda: client.get("/simulation/results"))

# 18. Adaptive Convergence
run_test("GET /adaptive/convergence", lambda: client.get("/adaptive/convergence"))

# 19. Adaptive Reset
run_test("POST /adaptive/reset", lambda: client.post("/adaptive/reset"))

print("\n" + "="*40)
if failed:
    print(f"Finished. {endpoints_tested} passed, {len(failed)} failed.")
    print(f"Failed endpoints: {failed}")
else:
    print(f"SUCCESS! All {endpoints_tested} endpoints tested passed.")
