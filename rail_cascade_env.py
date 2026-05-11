"""
RailCascade Mini V2 -- OpenEnv Reinforcement Learning Environment
Deterministic railway traffic management on a 10-node directed graph.

An AI agent coordinates multiple trains, issuing per-train hold/reroute/noop
actions each timestep to minimize total delay caused by blocked edges,
edge conflicts, and node-occupancy cascading.

Graph: 10 nodes (2 sources, 4 junctions, 3 corridors, 1 terminal)
       14 directed edges with 3 alternate bypass corridors
"""

from __future__ import annotations

import json
from collections import deque
from copy import deepcopy
from typing import Any, Optional, Literal

from pydantic import BaseModel, model_validator


# -------------------------------------------------------------
# Pydantic Models -- Typed API contracts
# -------------------------------------------------------------

class TrainState(BaseModel):
    """Observable state of a single train."""
    id: int
    position: str
    delay: int
    status: Literal["moving", "held", "blocked", "arrived", "stranded"]
    path_remaining: list[str] = []
    arrived: bool = False
    held: bool = False


class TrackState(BaseModel):
    """State of a single directed track."""
    id: str
    blocked: bool
    capacity: int


class ObservationState(BaseModel):
    """Full observable state returned by reset() and step()."""
    timestep: int
    max_steps: int
    task: str
    done: bool
    total_delay: int
    nodes: list[str]
    tracks: list[TrackState]
    trains: list[TrainState]


class SingleAction(BaseModel):
    """One action targeting one train."""
    train_id: int
    action: Literal["hold", "reroute", "noop"]
    new_path: Optional[list[str]] = None

    @model_validator(mode="after")
    def validate_action(self) -> "SingleAction":
        if self.action == "reroute":
            if not self.new_path or len(self.new_path) < 1:
                raise ValueError("reroute action requires a non-empty new_path")
        return self


class StepActions(BaseModel):
    """Container for all per-train actions in one timestep."""
    actions: list[SingleAction] = []


class StepReward(BaseModel):
    """Reward signal returned each timestep."""
    step_reward: float
    reward_breakdown: dict[str, float]
    total_delay: int
    final_score: Optional[float] = None


# -------------------------------------------------------------
# Internal Train -- Mutable runtime object
# -------------------------------------------------------------

class _Train:
    """Internal mutable train representation (not exposed via API)."""

    __slots__ = ("id", "position", "destination", "path", "delay", "held")

    def __init__(
        self,
        train_id: int,
        position: str,
        destination: str,
        path: list[str],
    ):
        self.id: int = train_id
        self.position: str = position
        self.destination: str = destination
        self.path: deque[str] = deque(path)
        self.delay: int = 0
        self.held: bool = False

    @property
    def arrived(self) -> bool:
        return self.position == self.destination and len(self.path) == 0

    @property
    def status(self) -> str:
        if self.arrived:
            return "arrived"
        if self.held:
            return "held"
        if len(self.path) == 0 and self.position != "T1":
            return "stranded"
        return "moving"

    def to_state(self, blocked=False) -> TrainState:
        return TrainState(
            id=self.id,
            position=self.position,
            delay=self.delay,
            status="blocked" if blocked else self.status,
            path_remaining=list(self.path),
            arrived=self.arrived,
            held=self.held,
        )


# -------------------------------------------------------------
# Graph Definition -- 10 nodes, 14 directed edges
# -------------------------------------------------------------

GRAPH_ADJACENCY: dict[str, list[str]] = {
    "S1": ["J1"],
    "S2": ["J4"],
    "J1": ["J2", "C1"],
    "J2": ["J3", "C2", "J4"],
    "J3": ["T1", "C3"],
    "J4": ["J3", "T1"],
    "C1": ["J2"],
    "C2": ["J3"],
    "C3": ["J4"],
    "T1": [],
}

ALL_NODES: list[str] = list(GRAPH_ADJACENCY.keys())

# Precompute all directed edges as (src, dst) tuples
ALL_EDGES: list[tuple[str, str]] = []
for _src, _dsts in GRAPH_ADJACENCY.items():
    for _dst in _dsts:
        ALL_EDGES.append((_src, _dst))

# Node type classification (for frontend rendering)
NODE_TYPES: dict[str, str] = {
    "S1": "source", "S2": "source",
    "J1": "junction", "J2": "junction", "J3": "junction", "J4": "junction",
    "C1": "corridor", "C2": "corridor", "C3": "corridor",
    "T1": "terminal",
}

# Node positions for visualization (x, y in canvas coordinates)
NODE_POSITIONS: dict[str, tuple[int, int]] = {
    "S1": (80, 300),
    "S2": (300, 60),
    "J1": (240, 300),
    "J2": (440, 300),
    "J3": (680, 300),
    "J4": (560, 120),
    "C1": (300, 460),
    "C2": (560, 460),
    "C3": (740, 120),
    "T1": (880, 250),
}

# Topological order (used for cascade propagation in Phase 4b)
# Furthest from T1 first
TOPOLOGICAL_ORDER: list[str] = [
    "S1", "S2", "C1", "J1", "C2", "C3", "J2", "J4", "J3", "T1"
]


# -------------------------------------------------------------
# Task Configurations
# -------------------------------------------------------------

TASK_CONFIGS: dict[str, dict[str, Any]] = {
    "easy": {
        "n_trains": 3,
        "start_positions": ["S1", "S1", "S2"],
        "blocked_edges": [("J2", "J3")],
        "max_steps": 20,
    },
    "medium": {
        "n_trains": 6,
        "start_positions": ["S1", "S1", "S1", "S2", "S2", "S1"],
        "blocked_edges": [("J2", "J3"), ("J1", "C1"), ("J4", "T1")],
        "max_steps": 30,
    },
    "hard": {
        "n_trains": 8,
        "start_positions": ["S1", "S1", "S1", "S1", "S1", "S2", "S2", "S2"],
        "blocked_edges": [("J2", "J3"), ("J4", "T1"), ("J1", "C1")],
        "max_steps": 40,
    },
    "dynamic_medium": {
        "n_trains": 8,
        "start_positions": ["S1", "S1", "S1", "S1", "S2", "S2", "S1", "S1"],
        "blocked_edges": [("J2", "J3")],
        "max_steps": 50,
        "dynamic_block_interval": 3,
        "dynamic_block_pool": [
            ("J4", "T1"), ("C2", "J3"), ("J1", "C1"), ("J3", "C3")
        ],
    },
    "extreme": {
        "n_trains": 10,
        "start_positions": ["S1", "S1", "S1", "S1", "S1", "S2", "S2", "S2", "S1", "S1"],
        "blocked_edges": [("J2", "J3"), ("J4", "T1")],
        "max_steps": 60,
        "dynamic_block_interval": 4,
        "dynamic_block_pool": [
            ("J1", "C1"), ("C2", "J3"), ("J3", "C3"), ("J2", "J4"), ("C1", "J2")
        ],
    },
    "vip_routing": {
        "n_trains": 6,
        "start_positions": ["S1", "S1", "S1", "S2", "S2", "S1"],
        "blocked_edges": [("J2", "J3"), ("J1", "C1")],
        "max_steps": 30,
        "objective": "vip_routing",
        "vip_trains": [0, 3],
    },
}



# -------------------------------------------------------------
# RailCascade Environment
# -------------------------------------------------------------

class RailCascadeEnv:
    """
    OpenEnv-compliant railway traffic management environment.

    API:
        reset()             -> ObservationState
        step(StepActions)   -> (ObservationState, StepReward, bool, dict)
        state()             -> dict
        get_score()         -> float

    Deterministic: identical inputs always produce identical outputs.
    """

    def __init__(self, task: str = "easy"):
        if task not in TASK_CONFIGS:
            raise ValueError(
                f"Unknown task '{task}'. Choose from: {list(TASK_CONFIGS.keys())}"
            )
        self.task: str = task
        self.config: dict[str, Any] = TASK_CONFIGS[task]
        self.n_trains: int = self.config["n_trains"]
        self.max_steps: int = self.config["max_steps"]

        # Mutable state -- populated by reset()
        self.trains: list[_Train] = []
        self.blocked_edges: set[tuple[str, str]] = set()
        self.timestep: int = 0
        self.total_delay: int = 0
        self.done: bool = False
        self._initialized: bool = False

        # Step-level tracking for reward computation
        self._step_new_delay: int = 0
        self._step_conflicts: int = 0

        # Dynamic blocking config
        self._dynamic_block_pool = list(self.config.get("dynamic_block_pool", []))
        self._dynamic_block_interval = self.config.get("dynamic_block_interval", 0)


    # --- Graph Helpers ---------------------------------------

    def _available_adjacency(self) -> dict[str, list[str]]:
        """Return adjacency list with blocked edges removed."""
        result: dict[str, list[str]] = {}
        for src, dsts in GRAPH_ADJACENCY.items():
            result[src] = [d for d in dsts if (src, d) not in self.blocked_edges]
        return result

    def _bfs_shortest_path(self, src: str, dst: str) -> list[str]:
        """
        BFS shortest path from src to dst on the directed graph,
        respecting currently blocked edges.

        Returns list of nodes from src to dst INCLUSIVE.
        Raises ValueError if no path exists.
        """
        if src == dst:
            return [src]

        adj = self._available_adjacency()
        visited: set[str] = {src}
        queue: deque[list[str]] = deque([[src]])

        while queue:
            path = queue.popleft()
            current = path[-1]
            for neighbor in adj.get(current, []):
                if neighbor in visited:
                    continue
                new_path = path + [neighbor]
                if neighbor == dst:
                    return new_path
                visited.add(neighbor)
                queue.append(new_path)

        raise ValueError(
            f"No path from '{src}' to '{dst}' with blocked edges "
            f"{list(self.blocked_edges)}"
        )

    def _validate_path(self, position: str, new_path: list[str]) -> bool:
        """
        Validate that new_path forms a legal route from position to T1.

        new_path should NOT include the current position.
        It must be a sequence of nodes where each consecutive pair
        (starting from position) forms an unblocked edge.
        """
        if not new_path:
            return False

        adj = self._available_adjacency()

        # Check: position -> new_path[0] is a valid edge
        full_sequence = [position] + new_path
        for i in range(len(full_sequence) - 1):
            if full_sequence[i + 1] not in adj.get(full_sequence[i], []):
                return False

        # Must end at T1
        if new_path[-1] != "T1":
            return False

        return True

    def _get_train(self, train_id: int) -> Optional[_Train]:
        """Look up a train by ID. Returns None if not found."""
        for t in self.trains:
            if t.id == train_id:
                return t
        return None

    # --- OpenEnv API -----------------------------------------

    def reset(self) -> ObservationState:
        """Initialize/reset the environment. Returns the initial observation."""
        self.timestep = 0
        self.total_delay = 0
        self.done = False
        self._initialized = True
        self._step_new_delay = 0
        self._step_conflicts = 0

        # Set blocked edges from config
        self.blocked_edges = set(
            tuple(e) for e in self.config["blocked_edges"]
        )

        # Create trains with BFS-computed shortest paths
        self.trains = []
        self.trajectory = []
        destination = "T1"
        start_positions = self.config["start_positions"]

        for i in range(self.n_trains):
            start = start_positions[i]
            full_path = self._bfs_shortest_path(start, destination)
            # path_remaining excludes the starting position
            path_remaining = full_path[1:]

            train = _Train(
                train_id=i,
                position=start,
                destination=destination,
                path=path_remaining,
            )
            self.trains.append(train)

        # Cache ONCE at start — never re-run BFS on final blocked state
        self._cached_sum_optimal = sum(
            len(self._bfs_shortest_path(s, "T1")) - 1
            for s in self.config["start_positions"]
        )
        obs = self._build_observation()
        self.trajectory.append({"obs": obs.model_dump()})
        return obs

    def step(
        self, actions: StepActions
    ) -> tuple[ObservationState, StepReward, bool, dict]:
        """
        Apply per-train actions and advance the simulation one timestep.

        Transition phases (executed in strict order):
            Phase 1:  Apply agent actions (hold/reroute/noop)
            Phase 2:  Collect movement intents
            Phase 3:  Blocked edge check
            Phase 4:  Edge conflict resolution (lowest ID wins)
            Phase 4b: Node occupancy blocking (cascade propagation)
            Phase 5:  Execute movement
            Phase 6:  Update timestep, check termination

        Returns (observation, reward, done, info).
        """
        if not self._initialized:
            raise RuntimeError("Must call reset() before step()")
        if self.done:
            raise RuntimeError(
                "Episode is done. Call reset() to start a new episode."
            )

        info: dict[str, Any] = {}
        new_delay = 0
        conflict_count = 0

        # -- Phase 1: Apply Actions --------------------------
        # Clear held flags from previous step
        for train in self.trains:
            train.held = False

        # Index actions by train_id for O(1) lookup
        action_map: dict[int, SingleAction] = {}
        valid_train_ids = {t.id for t in self.trains}
        for act in actions.actions:
            if act.train_id not in valid_train_ids:
                info.setdefault("invalid_actions", []).append(act.train_id)
            else:
                action_map[act.train_id] = act

        reroute_results: dict[int, bool] = {}

        for train in self.trains:
            if train.arrived:
                continue

            act = action_map.get(train.id)
            if act is None:
                # No action specified -> noop
                continue

            if act.action == "hold":
                train.held = True
                train.delay += 1
                new_delay += 1

            elif act.action == "reroute":
                new_path = act.new_path or []
                if self._validate_path(train.position, new_path):
                    train.path = deque(new_path)
                    reroute_results[train.id] = True
                else:
                    reroute_results[train.id] = False
            # 'noop' -> do nothing

        if reroute_results:
            info["reroute_results"] = reroute_results

        # -- Phase 2: Collect Movement Intents ---------------
        # Recover stranded trains (empty path at non-terminal node)
        for train in self.trains:
            if not train.arrived and len(train.path) == 0 and train.position != "T1":
                try:
                    full_path = self._bfs_shortest_path(train.position, "T1")
                    train.path = deque(full_path[1:])
                    info.setdefault("recovered_trains", []).append(train.id)
                except ValueError:
                    pass  # Truly unreachable -- leave stranded, will timeout

        # intent_list: list of (train_id, src, dst) for trains that want to move
        intent_set: dict[int, tuple[str, str]] = {}  # train_id -> (src, dst)

        for train in self.trains:
            if train.arrived or train.held:
                continue
            if len(train.path) == 0:
                continue
            next_node = train.path[0]
            intent_set[train.id] = (train.position, next_node)

        # -- Phase 3: Blocked Edge Check ---------------------
        blocked_in_phase3: list[int] = []
        for tid, (src, dst) in list(intent_set.items()):
            if (src, dst) in self.blocked_edges:
                train = self._get_train(tid)
                if train:
                    train.delay += 1
                    new_delay += 1
                blocked_in_phase3.append(tid)
                del intent_set[tid]

        # -- Phase 4: Edge Conflict Resolution ---------------
        # Group intents by edge (src, dst)
        edge_groups: dict[tuple[str, str], list[int]] = {}
        for tid, edge in intent_set.items():
            if edge not in edge_groups:
                edge_groups[edge] = []
            edge_groups[edge].append(tid)

        blocked_in_phase4: list[int] = []
        conflict_edges: list[dict] = []
        for edge, train_ids in edge_groups.items():
            if len(train_ids) > 1:
                train_ids.sort()  # Deterministic: lowest ID wins
                winner = train_ids[0]
                for loser_id in train_ids[1:]:
                    train = self._get_train(loser_id)
                    if train:
                        train.delay += 2      # conflict penalty = 2, hold = 1 -> holding is now worth it
                        new_delay += 2
                        conflict_count += 1
                    blocked_in_phase4.append(loser_id)
                    del intent_set[loser_id]
                conflict_edges.append({
                    "edge": list(edge),
                    "winner": winner,
                    "blocked": train_ids[1:],
                })

        # -- Phase 4b: Node Occupancy Blocking (Cascade) -----
        # Determine which nodes are occupied by stationary trains.
        # IMPORTANT: T1 (terminal) has UNLIMITED capacity — arrived trains
        # at T1 do NOT block other trains from entering.
        # Only intermediate nodes block when occupied by a stationary train.
        moving_train_ids = set(intent_set.keys())
        stationary_positions: dict[str, int] = {}  # node -> train_id occupying it

        for train in self.trains:
            if train.id not in moving_train_ids:
                # This train is stationary — but T1 has unlimited capacity
                if train.position != "T1" and train.position not in stationary_positions:
                    stationary_positions[train.position] = train.id

        # Process intents in reverse topological order for cascade propagation
        # Sort remaining intents by topological position of their SOURCE
        topo_index = {node: i for i, node in enumerate(TOPOLOGICAL_ORDER)}

        # We may need multiple passes if cascades chain.
        # Processing in topological order (furthest from T1 first) handles
        # single-pass cascade for most cases. We do max N passes for safety.
        cascade_blocked: list[int] = []
        changed = True
        max_cascade_passes = len(self.trains)
        pass_count = 0

        while changed and pass_count < max_cascade_passes:
            changed = False
            pass_count += 1

            for tid in sorted(
                list(intent_set.keys()),
                key=lambda t: topo_index.get(
                    intent_set[t][0] if t in intent_set else "T1", 999
                ),
            ):
                if tid not in intent_set:
                    continue
                _, target_node = intent_set[tid]

                if target_node in stationary_positions:
                    # Target node is occupied by a stationary train
                    train = self._get_train(tid)
                    if train:
                        train.delay += 2      # conflict penalty = 2, hold = 1 -> holding is now worth it
                        new_delay += 2
                        conflict_count += 1
                    cascade_blocked.append(tid)
                    # This train is now stationary too -- it might block others
                    if train.position != "T1":
                        stationary_positions[train.position] = train.id
                    del intent_set[tid]
                    changed = True


        # -- Phase 5: Execute Movement -----------------------
        moved_count = 0
        arrived_this_step = 0
        for tid in intent_set:
            train = self._get_train(tid)
            if train and len(train.path) > 0:
                next_node = train.path.popleft()
                train.position = next_node
                moved_count += 1
                if train.arrived:
                    arrived_this_step += 1

        # -- Phase 6: Update Timestep + Termination ----------
        self.timestep += 1
        self.total_delay = sum(t.delay for t in self.trains)

        all_arrived = all(t.arrived for t in self.trains)
        if all_arrived or self.timestep >= self.max_steps:
            self.done = True

        # -- Compute Reward ----------------------------------
        r_delay = -1.0 * float(new_delay)
        r_conflict = -2.0 * float(conflict_count)
        r_move = 0.1 * float(moved_count)
        r_arrival = 10.0 * float(arrived_this_step)
        
        step_reward = r_delay + r_conflict + r_move + r_arrival
        reward_breakdown = {
            "delay": r_delay,
            "conflict": r_conflict,
            "movement": r_move,
            "arrival": r_arrival
        }

        final_score = None
        if self.done:
            final_score = self.get_score()

        reward = StepReward(
            step_reward=step_reward,
            reward_breakdown=reward_breakdown,
            total_delay=self.total_delay,
            final_score=final_score,
        )

        self._step_new_delay = new_delay
        self._step_conflicts = conflict_count

        # Build info dict
        info["timestep"] = self.timestep
        info["total_delay"] = self.total_delay
        info["new_delay"] = new_delay
        info["conflicts"] = conflict_count
        if conflict_edges:
            info["conflict_edges"] = conflict_edges
        if blocked_in_phase3:
            info["blocked_by_edge"] = blocked_in_phase3
        if cascade_blocked:
            info["cascade_blocked"] = cascade_blocked

        # -- Phase 6: Dynamic Blocking (Post-Step) -----------
        if self._dynamic_block_interval and self.timestep > 0 and self.timestep % self._dynamic_block_interval == 0:
            available = [e for e in self._dynamic_block_pool 
                         if tuple(e) not in self.blocked_edges]
            if available:
                # Deterministic selection: use timestep as index
                new_block = tuple(available[self.timestep % len(available)])
                self.blocked_edges.add(new_block)
                info["new_block"] = list(new_block)
                
                # Reroute any trains whose path now crosses this blocked edge
                for train in self.trains:
                    if not train.arrived and len(train.path) > 0:
                        # Construct a list of edges in the current path
                        path_nodes = [train.position] + list(train.path)
                        path_edges = [(path_nodes[i], path_nodes[i+1]) for i in range(len(path_nodes)-1)]
                        
                        if new_block in path_edges:
                            try:
                                new_full = self._bfs_shortest_path(train.position, "T1")
                                train.path = deque(new_full[1:])
                            except ValueError:
                                # If no path exists, train stays on current path (will be blocked next step)
                                pass

        obs = self._build_observation(blocked_trains=set(blocked_in_phase3 + blocked_in_phase4 + cascade_blocked))
        
        self.trajectory.append({
            "actions": actions.model_dump(),
            "reward": reward.model_dump(),
            "obs": obs.model_dump(),
            "info": deepcopy(info)
        })

        return obs, reward, self.done, info

    def state(self) -> dict:
        """Return full serializable snapshot of the current environment state."""
        return self._build_observation().model_dump()

    def get_score(self) -> float:
        """
        Deterministic final grader.

        Two components:
        1. Delay score: 1.0 - total_delay / max_tolerable_delay
           max_tolerable_delay = n_trains * optimal_path_length
           where optimal_path_length is the average BFS shortest path length
           across all trains at reset time. This makes the denominator tight:
           if total delay equals the optimal travel distance, score = 0.

        2. Arrival penalty: unarrived trains reduce score proportionally.

        Final: score = delay_score * arrival_ratio

        Returns a float in [0.0, 1.0].
        """
        max_tolerable_delay = max(self._cached_sum_optimal, 1)

        # Apply VIP objective delay scaling
        if self.config.get("objective") == "vip_routing":
            vip_trains = set(self.config.get("vip_trains", []))
            calc_delay = sum(t.delay * (5 if t.id in vip_trains else 1) for t in self.trains)
        else:
            calc_delay = self.total_delay

        delay_score = max(0.0, 1.0 - calc_delay / max_tolerable_delay)
        arrived_count = sum(1 for t in self.trains if t.arrived)
        arrival_ratio = arrived_count / max(self.n_trains, 1)
        return max(0.0, min(1.0, delay_score * arrival_ratio))


    # --- Internal Helpers ------------------------------------

    def _build_observation(self, blocked_trains: set[int] = None) -> ObservationState:
        """Construct an ObservationState from current internal state."""
        if blocked_trains is None:
            blocked_trains = set()
            
        tracks = [
            TrackState(
                id=f"{src}->{dst}",
                blocked=(src, dst) in self.blocked_edges,
                capacity=999 if dst == "T1" else 1
            )
            for src, dst in ALL_EDGES
        ]
        return ObservationState(
            timestep=self.timestep,
            max_steps=self.max_steps,
            task=self.task,
            done=self.done,
            total_delay=self.total_delay,
            nodes=ALL_NODES,
            tracks=tracks,
            trains=[t.to_state(blocked=(t.id in blocked_trains)) for t in self.trains],
        )


# -------------------------------------------------------------
# Greedy Baseline Agent
# -------------------------------------------------------------

def greedy_agent(env: RailCascadeEnv) -> StepActions:
    
    """
    Simple greedy baseline agent.

    Strategy:
        For each non-arrived train (in ID order):
            - If the train's next edge is blocked -> reroute via BFS shortest path
            - Otherwise -> noop

    This agent NEVER holds trains. It only reacts to blocked edges.
    It cannot:
        1. Anticipate conflicts (no lookahead)
        2. Stagger train departures (no temporal planning)
        3. Balance load across corridors (always picks shortest path)
    """
    actions: list[SingleAction] = []

    for train in env.trains:
        if train.arrived:
            continue

        if len(train.path) == 0:
            continue

        next_node = train.path[0]
        edge = (train.position, next_node)

        if edge in env.blocked_edges:
            # Reroute via BFS
            try:
                full_path = env._bfs_shortest_path(train.position, "T1")
                new_remaining = full_path[1:]  # Exclude current position
                actions.append(SingleAction(
                    action="reroute",
                    train_id=train.id,
                    new_path=new_remaining,
                ))
            except ValueError:
                # No path available -- noop
                actions.append(SingleAction(
                    action="noop",
                    train_id=train.id,
                ))
        else:
            # No issue -- noop (let it move naturally)
            actions.append(SingleAction(
                action="noop",
                train_id=train.id,
            ))

    return StepActions(actions=actions)


# -------------------------------------------------------------
# Explicit Grader Functions
# -------------------------------------------------------------

def grade_trajectory(trajectory: list[dict], task_config: dict) -> float:
    """Core grader logic taking a full stateless trajectory."""
    if not trajectory:
        return 0.0
        
    final_step = trajectory[-1]
    final_reward = final_step.get("reward", {})
    if final_reward.get("final_score") is not None:
        return final_reward["final_score"]
    
    return 0.0

def grade_easy(trajectory: list[dict]) -> float:
    return grade_trajectory(trajectory, TASK_CONFIGS["easy"])

def grade_medium(trajectory: list[dict]) -> float:
    return grade_trajectory(trajectory, TASK_CONFIGS["medium"])

def grade_hard(trajectory: list[dict]) -> float:
    return grade_trajectory(trajectory, TASK_CONFIGS["hard"])

def grade_vip(trajectory: list[dict]) -> float:
    return grade_trajectory(trajectory, TASK_CONFIGS["vip_routing"])


# -------------------------------------------------------------
# Sanity Checks
# -------------------------------------------------------------

def run_sanity_checks() -> None:
    """Comprehensive sanity checks for all task levels."""
    print("=" * 70)
    print("RailCascade Mini V2 -- Sanity Checks")
    print("=" * 70)

    # -- Test 1: Graph connectivity --
    print("\n[TEST 1] Graph connectivity")
    env = RailCascadeEnv(task="easy")
    env.blocked_edges = set()
    for node in ALL_NODES:
        if node == "T1":
            continue
        try:
            path = env._bfs_shortest_path(node, "T1")
            print(f"  {node} -> T1: {' -> '.join(path)}")
        except ValueError as e:
            print(f"  FAIL: {e}")
            raise
    print("  [OK] All nodes can reach T1 with no blocked edges")

    # -- Test 2: Blocked edge rerouting --
    print("\n[TEST 2] Blocked edge rerouting")
    for task_name, config in TASK_CONFIGS.items():
        env = RailCascadeEnv(task=task_name)
        env.blocked_edges = set(tuple(e) for e in config["blocked_edges"])
        for start in set(config["start_positions"]):
            try:
                path = env._bfs_shortest_path(start, "T1")
                # Verify no edge in path is blocked
                for i in range(len(path) - 1):
                    edge = (path[i], path[i + 1])
                    assert edge not in env.blocked_edges, (
                        f"Path {path} uses blocked edge {edge}"
                    )
                print(f"  [{task_name}] {start} -> T1: {' -> '.join(path)}")
            except ValueError as e:
                print(f"  FAIL [{task_name}]: {e}")
                raise
    print("  [OK] All paths avoid blocked edges")

    # -- Test 3: Full episode with noop --
    print("\n[TEST 3] Noop-only episodes")
    for task_name in TASK_CONFIGS:
        env = RailCascadeEnv(task=task_name)
        obs = env.reset()
        steps = 0
        while not env.done:
            obs, reward, done, info = env.step(StepActions(actions=[]))
            steps += 1
        score = env.get_score()
        all_arrived = all(t.arrived for t in env.trains)
        print(f"  [{task_name}] steps={steps}, delay={env.total_delay}, "
              f"score={score:.4f}, all_arrived={all_arrived}")
        assert 0.0 <= score <= 1.0, f"Score {score} out of range!"
    print("  [OK] All noop episodes complete with valid scores")

    # -- Test 4: Full episode with greedy agent --
    print("\n[TEST 4] Greedy agent episodes")
    greedy_scores: dict[str, float] = {}
    for task_name in TASK_CONFIGS:
        env = RailCascadeEnv(task=task_name)
        obs = env.reset()
        steps = 0
        while not env.done:
            agent_actions = greedy_agent(env)
            obs, reward, done, info = env.step(agent_actions)
            steps += 1
        score = env.get_score()
        greedy_scores[task_name] = score
        all_arrived = all(t.arrived for t in env.trains)
        print(f"  [{task_name}] steps={steps}, delay={env.total_delay}, "
              f"score={score:.4f}, all_arrived={all_arrived}")
        assert 0.0 <= score <= 1.0, f"Score {score} out of range!"
    print("  [OK] All greedy episodes complete with valid scores")

    # -- Test 5: Determinism --
    print("\n[TEST 5] Determinism (greedy agent)")
    for task_name in TASK_CONFIGS:
        states_run1: list[dict] = []
        states_run2: list[dict] = []

        for run_states in [states_run1, states_run2]:
            env = RailCascadeEnv(task=task_name)
            env.reset()
            run_states.append(deepcopy(env.state()))
            while not env.done:
                agent_actions = greedy_agent(env)
                env.step(agent_actions)
                run_states.append(deepcopy(env.state()))

        assert len(states_run1) == len(states_run2), (
            f"[{task_name}] Run lengths differ: "
            f"{len(states_run1)} vs {len(states_run2)}"
        )
        for i, (s1, s2) in enumerate(zip(states_run1, states_run2)):
            assert s1 == s2, (
                f"[{task_name}] State divergence at step {i}"
            )
        print(f"  [{task_name}] [OK] Identical over {len(states_run1)} states")
    print("  [OK] All tasks are perfectly deterministic")

    # -- Test 6: Path validation --
    print("\n[TEST 6] Path validation")
    env = RailCascadeEnv(task="easy")
    env.reset()
    # Valid path
    assert env._validate_path("S1", ["J1", "J2", "C2", "J3", "T1"]) is True
    # Invalid: uses blocked edge
    assert env._validate_path("S1", ["J1", "J2", "J3", "T1"]) is False
    # Invalid: doesn't end at T1
    assert env._validate_path("S1", ["J1", "J2"]) is False
    # Invalid: disconnected
    assert env._validate_path("S1", ["J3", "T1"]) is False
    # Empty path
    assert env._validate_path("S1", []) is False
    print("  [OK] All path validation cases pass")

    # -- Test 7: Score formula bounds --
    print("\n[TEST 7] Score formula bounds")
    env = RailCascadeEnv(task="easy")
    env.reset()
    # Zero delay, all arrived -> score = 1.0
    env.total_delay = 0
    # All trains must be arrived for full score
    for t in env.trains:
        t.position = "T1"
        t.path = deque()
    score = env.get_score()
    assert score == 1.0, f"Expected 1.0, got {score}"
    # High delay -> score approaches 0.0
    env.total_delay = 99999
    score = env.get_score()
    assert score == 0.0, f"Expected 0.0, got {score}"
    # No trains arrived, zero delay -> score = 0.0 (arrival_ratio=0)
    env.total_delay = 0
    for t in env.trains:
        t.position = "S1"
        t.path = deque(["J1", "J2", "J4", "T1"])
    score = env.get_score()
    assert score == 0.0, f"Expected 0.0 (no arrivals), got {score}"
    print("  [OK] Score formula correctly bounded [0.0, 1.0]")

    # -- Test 8: Conflict detection --
    print("\n[TEST 8] Conflict detection")
    env = RailCascadeEnv(task="easy")
    env.reset()
    # Place two trains at J1 with same next hop
    env.trains[0].position = "J1"
    env.trains[0].path = deque(["J2", "C2", "J3", "T1"])
    env.trains[1].position = "J1"
    env.trains[1].path = deque(["J2", "C2", "J3", "T1"])
    obs, reward, done, info = env.step(StepActions(actions=[]))
    # Train 0 (lower ID) should have moved to J2
    # Train 1 should be blocked with delay = 1
    t0 = env._get_train(0)
    t1 = env._get_train(1)
    assert t0.position == "J2", f"Train 0 should be at J2, got {t0.position}"
    assert t1.position == "J1", f"Train 1 should be stuck at J1, got {t1.position}"
    assert t1.delay >= 1, f"Train 1 should have delay >= 1, got {t1.delay}"
    assert reward.reward_breakdown["conflict"] <= -2.0, (
        f"Should have at least 1 conflict, got {reward.reward_breakdown['conflict']}"
    )
    print(f"  Train 0: {t0.position} (moved), Train 1: {t1.position} (blocked)")
    print(f"  Conflicts: {reward.reward_breakdown['conflict'] / -2.0}")
    print("  [OK] Edge conflict detection works correctly")

    # -- Test 9: Cascade delay propagation --
    print("\n[TEST 9] Cascade delay propagation")
    env = RailCascadeEnv(task="easy")
    env.reset()
    # Set up a cascade: Train stuck at J2, Train 1 tries to move to J2
    env.trains[0].position = "J2"
    env.trains[0].path = deque(["J3", "T1"])
    env.trains[0].held = True  # Explicitly hold train 0

    env.trains[1].position = "J1"
    env.trains[1].path = deque(["J2", "C2", "J3", "T1"])

    env.trains[2].position = "S2"
    env.trains[2].path = deque(["J4", "J3", "T1"])

    # Step with hold on train 0
    hold_action = StepActions(actions=[
        SingleAction(action="hold", train_id=0),
    ])
    obs, reward, done, info = env.step(hold_action)

    t0 = env._get_train(0)
    t1 = env._get_train(1)
    # Train 0 is held at J2
    assert t0.position == "J2", f"Train 0 should be at J2, got {t0.position}"
    # Train 1 should be cascade-blocked (can't move to J2 because train 0 is there)
    assert t1.position == "J1", f"Train 1 should be stuck at J1, got {t1.position}"
    assert t1.delay >= 1, f"Train 1 should have cascade delay, got {t1.delay}"
    print(f"  Train 0: {t0.position} (held), delay={t0.delay}")
    print(f"  Train 1: {t1.position} (cascade blocked), delay={t1.delay}")
    print("  [OK] Cascade delay propagation works correctly")

    # -- Summary --
    print(f"\n{'=' * 70}")
    print("ALL SANITY CHECKS PASSED [OK]")
    print(f"{'=' * 70}")

    # Print score comparison
    print("\nGreedy Agent Score Summary:")
    print(f"  {'Task':<10} {'Score':>8}")
    print(f"  {'-' * 20}")
    for task_name, score in greedy_scores.items():
        print(f"  {task_name:<10} {score:>8.4f}")


# -------------------------------------------------------------
# Example Run Loop
# -------------------------------------------------------------

def example_run(task: str = "easy", verbose: bool = True) -> float:
    """
    Run a complete episode with the greedy agent.
    Returns the final score.
    """
    env = RailCascadeEnv(task=task)
    obs = env.reset()

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"  Example Run -- Task: {task}")
        print(f"  Trains: {env.n_trains}, Horizon: {env.max_steps}")
        print(f"  Blocked: {list(env.blocked_edges)}")
        print(f"{'=' * 50}")
        print(f"\nInitial positions:")
        for t in obs.trains:
            print(f"  Train {t.id}: {t.position} -> {t.destination}, "
                  f"path={t.path_remaining}")

    step_count = 0
    while not env.done:
        agent_actions = greedy_agent(env)
        obs, reward, done, info = env.step(agent_actions)
        step_count += 1

        if verbose:
            positions = ", ".join(
                f"T{t.id}@{t.position}"
                + ("[OK]" if t.arrived else "")
                for t in obs.trains
            )
            print(f"  t={obs.timestep:>3d} | {positions} | "
                  f"delay={reward.total_delay}, "
                  f"conflicts={reward.reward_breakdown['conflict'] / -2.0}, "
                  f"r={reward.step_reward:.1f}")
            
            if "new_block" in info:
                print(f"  >>> DYNAMIC BLOCK EVENT: {info['new_block'][0]}->{info['new_block'][1]} blocked!")


    final_score = env.get_score()
    if verbose:
        print(f"\n  Final: steps={step_count}, total_delay={env.total_delay}, "
              f"score={final_score:.4f}")
        arrived = sum(1 for t in env.trains if t.arrived)
        print(f"  Arrived: {arrived}/{env.n_trains}")

    return final_score


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "run":
        # Run example episodes
        task = sys.argv[2] if len(sys.argv) > 2 else "all"
        if task == "all":
            for t in ["easy", "medium", "hard", "dynamic_medium", "extreme"]:
                example_run(t, verbose=True)
        else:
            example_run(task, verbose=True)
    else:
        # Run sanity checks
        run_sanity_checks()
