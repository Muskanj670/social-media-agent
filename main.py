from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict

from my_agent.agent import run_agent

app = FastAPI()


class RunRequest(BaseModel):
    task_type: str
    payload: Dict[str, Any] = {}


@app.get("/")
def home():
    return {"message": "AI Agent Running 🚀"}


@app.post("/run")
def run(request: RunRequest):
    return run_agent(request.task_type, request.payload)
