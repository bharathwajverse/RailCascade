# RailCascade Mini V2: Project Vision & Architecture

## 1. How to Use the Project

RailCascade Mini V2 is an OpenEnv-compliant reinforcement learning environment designed for railway traffic management. The project is split into two primary modes of operation: headless simulation for agent training, and an interactive frontend for human visualization and manual testing.

### Basic Setup
```bash
pip install -r requirements.txt
```

### Running Simulations (Headless/RL Training)
The core logic resides entirely in `rail_cascade_env.py`. You can import `RailCascadeEnv` into your RL training loops (e.g., using Ray RLlib, Stable Baselines3, or custom agent scripts).

```python
from rail_cascade_env import RailCascadeEnv, SingleAction, StepActions

env = RailCascadeEnv(task="hard") # Options: easy, medium, hard
obs = env.reset()

while not env.done:
    # Example: issue a "noop" to all trains
    actions = StepActions(actions=[
        SingleAction(action_type="noop", train_id=t.id) for t in env.trains
    ])
    obs, reward, done, info = env.step(actions)
    
print(f"Final Score: {env.get_score()}")
```

### Interactive Dashboard (Visualization)
To visualize the environment, run the FastAPI server:
```bash
python server.py
```
Open `http://localhost:8000` in your browser. The dashboard allows you to select tasks, reset the environment, step through the simulation manually, or use the "Auto" feature to watch the greedy baseline agent attempt the routing challenge.

---

## 2. Why It Is Good

RailCascade Mini V2 stands out because it solves the fundamental issues present in generic grid-world routing simulations by accurately capturing real-world railway physics and constraints.

*   **Explicit Delay Cascades (Phase 4b Logic):** Unlike typical grid-worlds where blocked agents simply wait without consequence, this environment tracks *occupancy*. If Train A is blocked, it occupies its current node. If Train B wants to move into Train A's node, Train B is also blocked (cascade). This perfectly models real-world railway network fragility.
*   **Edge-Based Conflict Resolution:** Tracks capacity at the edge level. Edge capacity is strictly 1. If two trains compete, conflict resolution is deterministic (lowest ID wins), making the environment entirely predictable and solvable without fighting RNG (Random Number Generators).
*   **Per-Train Granular Control:** The agent issues localized, sparse actions (`hold`, `reroute`, `noop`) rather than single global actions, perfectly aligning with Multi-Agent Reinforcement Learning (MARL) paradigms.
*   **Anti-Gaming Reward Structure:** The reward function penalizes delays but heavily multiplies conflict penalties (`-2.0` per conflict). The final grader (`score = delay_score * arrival_ratio`) tightly evaluates against the *optimal theoretical travel time*, ensuring that the baseline greedy agent fundamentally fails on harder tasks.
*   **Premium Glassmorphism Dashboard:** The built-in Canvas/FastAPI frontend allows researchers and stakeholders to visually debug policies in real-time, complete with transition interpolations and conflict logging.

---

## 3. How It Can Be Used in the Future

The environment is designed to serve as a benchmark for optimization and AI research in logistics:

*   **Multi-Agent Reinforcement Learning (MARL) Benchmark:** Use it to test Q-learning, PPO, or specialized MARL algorithms. Can decentralized agents learn to stagger departures to avoid bottlenecks at `T1`?
*   **LLM "Thinking" Agents:** Pass the JSON `state()` to an LLM. Test whether advanced language models can perform path-planning and multi-step lookahead to prevent cascading delays.
*   **Operations Research & Constraint Programming:** Compare traditional algorithmic solvers (like Dijkstra with time-windows or Mixed Integer Linear Programming) against RL agents on the same topology.
*   **Educational Tool:** Use the visual dashboard to teach students the complexities of network flow, congestion, and the "butterfly effect" of a single blocked track in transportation networks.

---

## 4. How to Implement It in the Future

To expand this from a "Mini" benchmark to a production-scale system:

1.  **OpenAI Gym/Gymnasium Wrapper:** Create a lightweight shim that wraps `rail_cascade_env.py` to match the strict `gymnasium.Env` interface (mapping observations to `Box`/`Discrete` spaces).
2.  **Vectorized Environments:** For RL training, implement a vectorized version (using `multiprocessing` or `jax`) to simulate thousands of cascades per second to gather training data efficiently.
3.  **Real-World Telemetry Ingestion (Digital Twin):** Instead of static tasks, feed real JSON transit data into `RailCascadeEnv`. Map actual city train stations to the topology, using the engine to predict delays 30 minutes into the future.
4.  **Hugging Face Spaces Deployment:** The current Dockerfile is ready. Build and deploy the FastAPI container to run the dashboard publicly.

---

## 5. What Improvements We Can Add

As the project evolves, the following features will elevate the environment:

*   **Dynamic/Stochastic Blocking:** Currently, blocked edges are static at `reset()`. Adding random breakdowns during an episode (e.g., edge `C2->J3` blocks at timestep 15) would test an agent's real-time adaptability.
*   **Variable Train Speeds & Lengths:** Currently, all trains move 1 edge per timestep. Real trains have varying velocities and lengths (occupying multiple nodes simultaneously).
*   **Two-Way Edges & Passing Loops:** Implement bi-directional tracks with sidings. This introduces the classic "deadlock" problem where two trains moving in opposite directions must negotiate who pulls into a passing loop.
*   **Fuel/Energy Costs in Reward:** Add a penalty for stopping and starting (momentum loss). A train that maintains a slow, steady speed is more efficient than one that stops completely and accelerates again.
*   **Continuous Observation Spaces:** While currently discrete, expanding node coordinates into continuous distances (e.g., Train 1 is 40% along Edge J2->J3) would allow integration with continuous control algorithms like SAC or DDPG.
*   **Agentic Inference Script:** Enhance `inference.py` to support OpenAI Swarm or multi-agent conversations, where one LLM controls Train A and another controls Train B, negotiating right-of-way.
