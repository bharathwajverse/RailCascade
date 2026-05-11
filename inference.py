import json
import os
import sys
import random
import numpy as np

from fastapi import FastAPI, Request
import uvicorn

# -------------------- DETERMINISM --------------------
random.seed(42)
np.random.seed(42)

# -------------------- IMPORT ENV ---------------------
# Use abspath to handle edge cases where __file__ resolves to '' on import
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from rail_cascade_env import RailCascadeEnv, StepActions, SingleAction

# -------------------- APP ----------------------------
app = FastAPI()

# -------------------- ROOT ---------------------------
@app.get("/")
async def root():
    return {"message": "RailCascade API is running", "status": "online"}

# -------------------- GLOBAL STATE ------------------
# The evaluator calls /reset then /step repeatedly in sequence.
# We must persist the env instance across requests — creating a new
# env on every call loses all simulation state.
_env: RailCascadeEnv | None = None

# -------------------- RESET --------------------------
@app.post("/reset")
@app.get("/reset")
async def reset_endpoint(request: Request):
    print("🔥 RESET HIT 🔥")
    global _env

    try:
        try:
            body = await request.json()
        except Exception:
            body = {}

        task = body.get("task", "medium") if isinstance(body, dict) else "medium"

        valid_tasks = ("easy", "medium", "hard", "dynamic_medium", "extreme", "vip_routing")
        if task not in valid_tasks:
            task = "medium"

        _env = RailCascadeEnv(task=task)
        obs = _env.reset()

        return {
            "state": json.loads(json.dumps(obs.model_dump(), default=str)),
            "reward": 0.0,
            "done": False,
            "info": {}
        }

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "state": None,
            "reward": 0.0,
            "done": True,
            "info": {}
        }

# -------------------- STEP ---------------------------
@app.post("/step")
async def step_endpoint(request: Request):
    print("🔥 STEP HIT 🔥")
    global _env

    try:
        # If no env exists yet (evaluator skipped /reset), initialise a default one
        if _env is None:
            _env = RailCascadeEnv(task="medium")
            _env.reset()

        # Parse action payload from request body
        try:
            body = await request.json()
        except Exception:
            body = {}

        # Build StepActions from the request body.
        # Evaluator may send: {"actions": [...]} or a bare list, or an empty body.
        if isinstance(body, dict) and "actions" in body:
            raw_actions = body["actions"]
        elif isinstance(body, list):
            raw_actions = body
        else:
            raw_actions = []

        # Parse each individual action safely
        parsed_actions = []
        for item in raw_actions:
            try:
                parsed_actions.append(SingleAction(**item))
            except Exception:
                # Skip malformed action entries rather than crashing
                continue

        step_actions = StepActions(actions=parsed_actions)

        # Execute the simulation step
        obs, reward, done, info = _env.step(step_actions)

        # Serialise reward (Pydantic model)
        try:
            reward_dict = reward.model_dump()
        except Exception:
            reward_dict = {"step_reward": 0.0, "reward_breakdown": {}, "total_delay": 0}

        return {
            "state": json.loads(json.dumps(obs.model_dump(), default=str)),
            "reward": json.loads(json.dumps(reward_dict, default=str)),
            "done": bool(done),
            "info": info if isinstance(info, dict) else {}
        }

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "state": None,
            "reward": 0.0,
            "done": True,
            "info": {}
        }

# -------------------- HEALTH -------------------------
@app.get("/ping")
async def ping():
    return {"status": "ok"}

# -------------------- LLM AGENT ----------------------
# OpenEnv injects API_BASE_URL and API_KEY into the environment.
# We MUST use these — never hardcode keys or use other providers.
def get_llm_actions(obs, task: str) -> StepActions:
    """Call the OpenEnv LiteLLM proxy to decide actions for this step."""
    import openai

    api_base = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    api_key  = os.environ.get("API_KEY", "")

    client = openai.OpenAI(base_url=api_base, api_key=api_key)

    # Build a compact state summary for the prompt
    active_trains = [t for t in obs.trains if not t.arrived]
    blocked_tracks = [t.id for t in obs.tracks if t.blocked]

    train_lines = []
    for t in active_trains:
        path_str = " -> ".join(t.path_remaining) if t.path_remaining else "none"
        train_lines.append(
            f"  Train {t.id}: pos={t.position} status={t.status} "
            f"delay={t.delay} path={path_str}"
        )

    prompt = f"""You are a railway traffic controller. Minimize total train delay.

Task: {task}
Step: {obs.timestep}/{obs.max_steps}
Blocked tracks: {blocked_tracks if blocked_tracks else 'none'}
Active trains:
{chr(10).join(train_lines) if train_lines else '  none'}

For each active train, choose ONE action:
- noop: continue on current path
- hold: stop the train for one step to avoid conflict
- reroute: provide a new path list (only if a clear alternative exists)

Respond ONLY with a valid JSON array. Each element must have:
  {{"train_id": <int>, "action": "noop"|"hold"|"reroute", "new_path": [<nodes>] or null}}

Rules:
- Use "reroute" only when the current path is fully blocked and you know an alternative.
- new_path must start from the train's NEXT node (not current position).
- For "noop" and "hold", set new_path to null.
- Do not include arrived trains.

JSON array only, no explanation:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        action_list = json.loads(raw)

        parsed = []
        for item in action_list:
            try:
                parsed.append(SingleAction(
                    train_id=int(item["train_id"]),
                    action=item["action"],
                    new_path=item.get("new_path") or None,
                ))
            except Exception:
                continue
        return StepActions(actions=parsed)

    except Exception as e:
        print(f"[LLM ERROR] {e} — falling back to noop", flush=True)
        return StepActions(actions=[])


# -------------------- STDOUT EVALUATION LOOP ---------
# OpenEnv requires [START]/[STEP]/[END] blocks printed to stdout.
# This runs ONLY when the file is executed directly (python inference.py).
# The FastAPI server is started separately via Dockerfile CMD — these
# two modes never run simultaneously.
def run_evaluation_loop(task: str = "medium") -> None:
    """Run one full episode with LLM agent and emit structured stdout blocks."""
    env = RailCascadeEnv(task=task)
    obs = env.reset()

    print(f"[START] task={task}", flush=True)

    step_num = 0
    done = False

    while not done:
        step_num += 1
        actions = get_llm_actions(obs, task)
        obs, reward, done, info = env.step(actions)

        step_reward = reward.step_reward if hasattr(reward, "step_reward") else 0.0
        print(f"[STEP] step={step_num} reward={round(step_reward, 4)}", flush=True)

    final_score = env.get_score()
    print(f"[END] task={task} score={round(final_score, 4)} steps={step_num}", flush=True)


# -------------------- MAIN ---------------------------
# Evaluator runs `python inference.py` directly and reads stdout blocks.
# The actual API server is started by Dockerfile CMD via uvicorn CLI.
if __name__ == "__main__":
    task = os.environ.get("TASK", "medium")
    run_evaluation_loop(task=task)