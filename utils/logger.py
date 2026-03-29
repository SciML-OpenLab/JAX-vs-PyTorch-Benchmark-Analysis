import json
import os
from datetime import datetime
import platform
import torch

def get_system_info():
    info = {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "processor": platform.processor(),
        "python_version": platform.python_version()
    }

    # GPU info (if available)
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
    else:
        info["gpu"] = "CPU"

    return info

def save_results(path, data, config=None, append=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "system": get_system_info(),
        "config": config,
        "results": data
    }

    # If append mode → keep history
    if append and os.path.exists(path):
        with open(path, "r") as f:
            existing = json.load(f)

        if not isinstance(existing, list):
            existing = [existing]

        existing.append(record)

        with open(path, "w") as f:
            json.dump(existing, f, indent=4)

    else:
        with open(path, "w") as f:
            json.dump(record, f, indent=4)

    print(f"Results saved to {path}")
