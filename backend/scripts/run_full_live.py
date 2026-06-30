import time
import json
import httpx
from datetime import datetime

BASE = "http://127.0.0.1:8000/v1"
PROMPT = "Decide whether to pursue a career transition into data science within the next 3 months; weigh options, timelines, and risks in a way that can produce a 12-node decision tree."

client = httpx.Client(timeout=60.0)

def post_json(path, payload):
    r = client.post(BASE + path, json=payload)
    r.raise_for_status()
    return r.json()

def get_json(path):
    for i in range(5):
        try:
            r = client.get(BASE + path)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            print(f"get_json failed ({exc}), retrying...")
            time.sleep(2)
    raise Exception(f"get_json failed repeatedly for {path}")

print("Creating mock intent...")
intent = post_json("/intake/mock-build-intent", {"prompt": PROMPT})
print("Intent created:", intent["id"])

# Update todo via print
print("Submitting ingestion job (scraping)...")
job = post_json("/knowledge/ingest", {"query": PROMPT})
job_id = job["job_id"]
print("Ingestion job id:", job_id)

# Poll ingestion job
for _ in range(120):
    status = get_json(f"/jobs/{job_id}")
    print("Ingest status:", status["status"])
    if status["status"] in ("completed", "failed"):
        break
    time.sleep(2)

if status["status"] != "completed":
    raise SystemExit(f'Ingestion did not complete successfully, status={status["status"]}')
else:
    print("Ingestion completed")

# Start simulation
print("Starting simulation (depth=2, branching=3, target=12 nodes)...")
sim = post_json("/simulate/start", {"user_intent_id": intent["id"], "disable_scraping": False, "depth": 2, "branching_factor": 3})
print("Simulation job id:", sim["job_id"])

sim_job_id = sim["job_id"]
for i in range(600):
    try:
        sj = get_json(f"/simulate/jobs/{sim_job_id}")
    except Exception as exc:
        print("Error fetching sim job:", exc)
        time.sleep(2)
        continue
    print("Sim status:", sj["status"], "progress:", sj.get("progress"))
    if sj["status"] in ("completed", "failed"):
        break
    time.sleep(3)

if sj["status"] != "completed":
    print("Simulation failed or timed out", sj.get("error_message"))
else:
    print("Simulation completed, fetching graph...")
    graph = get_json(f"/graph/{intent['id']}")
    now = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = f"run_output_{now}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    print("Saved graph to", out_path)
    print("Nodes:", len(graph.get("nodes", [])), "Edges:", len(graph.get("edges", [])))

print("Done")
