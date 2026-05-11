import os
import uvicorn
from inference import app

# -------------------- /grader ENDPOINT -------------------
# Some hackathon evaluators check for a /grader HTTP endpoint
# in addition to the grader block in openenv.yaml.
# This returns the grader spec for all 6 tasks.
from fastapi import FastAPI

@app.get("/grader")
@app.post("/grader")
async def grader_endpoint():
    return {
        "graders": [
            {"task": "easy",          "metric": "final_score", "range": [0.0, 1.0], "goal": "maximize", "passing_threshold": 0.5},
            {"task": "medium",        "metric": "final_score", "range": [0.0, 1.0], "goal": "maximize", "passing_threshold": 0.4},
            {"task": "hard",          "metric": "final_score", "range": [0.0, 1.0], "goal": "maximize", "passing_threshold": 0.3},
            {"task": "dynamic_medium","metric": "final_score", "range": [0.0, 1.0], "goal": "maximize", "passing_threshold": 0.3},
            {"task": "extreme",       "metric": "final_score", "range": [0.0, 1.0], "goal": "maximize", "passing_threshold": 0.2},
            {"task": "vip_routing",   "metric": "final_score", "range": [0.0, 1.0], "goal": "maximize", "passing_threshold": 0.3},
        ]
    }

# -------------------- MAIN -------------------------------
def main():
    """Entry point for the OpenEnv multi-mode deployment."""
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()