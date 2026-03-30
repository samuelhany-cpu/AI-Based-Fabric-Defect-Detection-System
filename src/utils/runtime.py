from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def configure_runtime_environment(config: dict[str, Any]) -> dict[str, str]:
    runtime_cfg = config.get("runtime", {})
    cache_root = Path(runtime_cfg.get("cache_root", "F:/Machine_Learning/.runtime-cache"))
    temp_dir = Path(runtime_cfg.get("temp_dir", cache_root / "tmp"))
    torch_home = Path(runtime_cfg.get("torch_home", cache_root / "torch"))
    mpl_config_dir = Path(runtime_cfg.get("mpl_config_dir", cache_root / "matplotlib"))

    for path in (cache_root, temp_dir, torch_home, mpl_config_dir):
        path.mkdir(parents=True, exist_ok=True)

    env_updates = {
        "TMP": str(temp_dir),
        "TEMP": str(temp_dir),
        "TMPDIR": str(temp_dir),
        "TORCH_HOME": str(torch_home),
        "MPLCONFIGDIR": str(mpl_config_dir),
    }
    for key, value in env_updates.items():
        os.environ[key] = value

    return env_updates


def describe_torch_runtime(torch_module) -> dict[str, object]:
    cuda_available = bool(torch_module.cuda.is_available())
    summary: dict[str, object] = {
        "torch_version": torch_module.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch_module.version.cuda,
        "device": "cpu",
    }

    if cuda_available:
        current_device = torch_module.cuda.current_device()
        summary["device"] = torch_module.cuda.get_device_name(current_device)
        summary["device_count"] = torch_module.cuda.device_count()
    else:
        summary["device_count"] = 0

    return summary
