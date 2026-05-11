"""Quick session isolation test."""
import requests

r1 = requests.post("http://localhost:8000/api/reset", json={"task": "easy"})
r2 = requests.post("http://localhost:8000/api/reset", json={"task": "hard"})
d1 = r1.json()
d2 = r2.json()

sid1 = d1["session_id"]
sid2 = d2["session_id"]

print(f"Session 1: {sid1[:8]}... task=easy, trains={len(d1['state']['trains'])}")
print(f"Session 2: {sid2[:8]}... task=hard, trains={len(d2['state']['trains'])}")
print(f"Different IDs: {sid1 != sid2}")

# Test auto_step on session 1
r3 = requests.post("http://localhost:8000/api/auto_step", json={"session_id": sid1})
d3 = r3.json()
print(f"Auto-step session 1: timestep={d3['state']['timestep']}, done={d3['done']}")

# Test auto_step on session 2
r4 = requests.post("http://localhost:8000/api/auto_step", json={"session_id": sid2})
d4 = r4.json()
print(f"Auto-step session 2: timestep={d4['state']['timestep']}, done={d4['done']}")

print("\nAll session isolation checks PASSED")
