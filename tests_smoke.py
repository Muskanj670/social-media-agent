from my_agent.agent import run_agent
import json

print(json.dumps(run_agent("text", {"prompt": "Hello world"}), indent=2))
print(json.dumps(run_agent("report", {"topic": "AI safety"}), indent=2))
print(json.dumps(run_agent("image", {"prompt": "Cat wearing a specs", "width": 256, "height": 256}), indent=2))
print(json.dumps(run_agent("video", {"prompt": "Walking cat", "width": 256, "height": 256, "frames": 8}), indent=2))
