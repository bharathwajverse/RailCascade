# RailCascade Mini V2: Vision & Future Roadmap

This document outlines the purpose, usage, improvements, and future potential of the **RailCascade Mini V2** project. It serves as a guide for reviewers, developers, and researchers looking to understand the value of this environment.

---

## 🚀 How to Use the Project

RailCascade Mini V2 is designed to be an **OpenEnv-compliant reinforcement learning environment** as well as an interactive web simulation.

### 1. Interactive Dashboard (Human / Visual Evaluation)
Start the FastAPI server to use the interactive frontend:
```bash
python server.py
```
* **Use Case:** Manually evaluate tasks like `easy`, `medium`, `hard`, `dynamic_medium`, and `extreme`. Observe the cascading delays, dynamic blocking, and real-time train conflict resolutions. 
* **Who it's for:** Traffic control operators, developers debugging agent behaviors, and hackathon judges verifying the logic.

### 2. Automated Benchmarking (LLM Agents)
Run the Gemini 2.0 Flash agent against the full task suite:
```bash
export HF_TOKEN="your-gemini-api-key"
python inference.py benchmark
```
* **Use Case:** Programmatically evaluate an LLM's spatial reasoning and multi-step planning capabilities. Logs are outputted in a structured JSONL format compliant with OpenEnv evaluation standards.

### 3. RL Training (Code Level)
Import the environment into any standard Python script to train PPO, DQN, or custom heuristics:
```python
from rail_cascade_env import RailCascadeEnv, StepActions, SingleAction

env = RailCascadeEnv(task="hard")
state = env.reset()
# ... loop step() function ...
```

---

## 🏆 Why It Is Good & How It Beats V1

RailCascade Mini V2 is a **massive leap** forward from the V1 prototype, shifting from a basic simulation to a **production-grade, deterministic grading environment**.

### Key Advantages Over V1
1. **Strict Determinism:** V1 relied on simple step logic. V2 implements a robust **6-Phase Transition Engine** (Actions -> Intents -> Block Checks -> Edge Conflicts -> Cascades -> Movement). This guarantees exactly the same result for the same actions, making it viable for strict RL training.
2. **Anti-Gaming Scoring:** In V1, models could "game" the system. V2 calculates max tolerable delay based on the sum of optimal BFS paths. If an agent deviates wildly, its score approaches mathematically bounded zero. 
3. **Session-based Concurrency:** V1 used a global environment state in the backend. V2 utilizes `session_id` isolation, meaning thousands of RL episodes or concurrent human viewers can interact with the API simultaneously without state corruption.
4. **LLM Lookahead Forecasting:** The V2 LLM prompt builder automatically calculates 1-step lookahead conflicts and explicitly warns the Gemini model of imminent collisions, pushing the LLM to think like a dynamic constraints solver.
5. **Dynamic Obstacles:** Introduction of the `dynamic_medium` and `extreme` tasks where edges are dynamically blocked mid-episode. This tests an agent's real-time adaptivity, a critical requirement for real-world traffic management.

---

## 🔮 Future Implementation & Real-World Use Cases

While currently a mini-graph representation, the architecture of RailCascade V2 is highly scalable. 

### Real-World Applications
* **Freight and Rail Dispatching:** The conflict resolution logic directly mirrors localized real-world railway interlocking systems. This codebase could be adapter for regional rail networks to suggest rerouting when a track segment is closed for maintenance.
* **Network Graph Routing:** The core logic isn't strictly limited to trains. It can be applied to data packet routing in mesh networks where nodes have capacity constraints and links fail dynamically.
* **AGV (Automated Guided Vehicles) in Warehouses:** The grid/graph logic applies perfectly to Amazon-style warehouse robots navigating tight corridors with single-lane constraints.

---

## 🛠️ Future Improvements

To take RailCascade to V3 and beyond, the following improvements are recommended:

1. **Bi-Directional Edges & Deadlocks:**
   Currently, the graph uses strict directed edges. Introducing bidirectional tracks where trains can face head-on deadlocks would drastically increase the difficulty and realism of the environment.
2. **Train Speed & Length Variations:**
   Presently, all trains move 1 edge per timestep and occupy 1 node. Future versions could introduce high-speed trains (moving 2 edges/step) and freight trains (occupying 2 consecutive nodes).
3. **PPO / DQN Baselines:**
   While the LLM agent and Greedy agent exist, integrating a fully trained Proximal Policy Optimization (PPO) agent using Stable Baselines3 would provide the ultimate benchmark comparison.
4. **Variable Node Capacities:**
   Allowing junctions (e.g., stations) to hold a queue of 2-3 trains simultaneously before cascading delays to upstream track segments.
