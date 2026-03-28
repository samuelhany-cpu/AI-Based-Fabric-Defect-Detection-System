from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_parent_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def write_json(data: dict[str, Any], output_path: str | Path) -> None:
    path = ensure_parent_dir(output_path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
