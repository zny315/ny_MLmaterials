from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


REQUIRED_SECTIONS = {
    "targets",
    "split_scope",
    "benchmark",
    "selected_backbones",
    "estimators",
    "pooling",
    "positive_affine",
    "synthesis",
}


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    missing = REQUIRED_SECTIONS.difference(config)
    if missing:
        raise ConfigError(f"configuration missing sections: {sorted(missing)}")
    targets = config["targets"]
    if not targets or any(task not in {"reg", "clf"} for task in targets.values()):
        raise ConfigError("targets must map endpoint names to 'reg' or 'clf'")
    scope = config["split_scope"]
    if not scope.get("N"):
        raise ConfigError("split_scope.N must contain at least one shot count")
    trials = scope.get("trials")
    if not isinstance(trials, list) or len(trials) != 2 or trials[0] > trials[1]:
        raise ConfigError("split_scope.trials must be an inclusive [first, last] range")
    for target, selection in config["selected_backbones"].items():
        if targets.get(target) != "reg":
            raise ConfigError(f"selected backbone target must be regression: {target}")
        if set(selection) != {"lab", "literature"}:
            raise ConfigError(f"selected_backbones[{target}] needs lab and literature")
    return config
