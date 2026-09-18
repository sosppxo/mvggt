import subprocess
from typing import Any

import torch


def detect_runtime_options() -> dict[str, Any]:
    local_gpu_available = False
    gpu_idle = False
    gpu_name = None
    reason = "no_gpu"

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        local_gpu_available = True
        gpu_name = torch.cuda.get_device_name(0)
        reason = "gpu_available"

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                first = result.stdout.strip().splitlines()[0]
                util, mem_used, mem_total = [float(x.strip()) for x in first.split(",")[:3]]
                mem_ratio = (mem_used / max(mem_total, 1.0))
                gpu_idle = util < 20 and mem_ratio < 0.6
                reason = "gpu_idle" if gpu_idle else "gpu_busy"
            else:
                gpu_idle = True
                reason = "gpu_available_no_smi"
        except Exception:
            gpu_idle = True
            reason = "gpu_available_no_smi"

    recommended_backend = "local" if local_gpu_available and gpu_idle else "hf_api"

    return {
        "local_gpu_available": local_gpu_available,
        "gpu_idle": gpu_idle,
        "gpu_name": gpu_name,
        "recommended_backend": recommended_backend,
        "reason": reason,
    }
