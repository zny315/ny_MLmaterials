from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy
import pandas
import scipy
import sklearn


def prepare_output(path: str | Path, *, overwrite: bool) -> Path:
    path = Path(path)
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(payload: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def run_metadata(config_path: Path, data_path: Path, command: str) -> dict[str, Any]:
    inputs = {}
    for name in ("features.npz", "source_targets.csv", "source_targets.parquet",
                 "target_samples.csv", "target_samples.parquet",
                 "split_manifest.csv", "split_manifest.parquet"):
        candidate = data_path / name
        if candidate.exists():
            inputs[name] = sha256_file(candidate)
    return {
        "command": command,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "data_path": str(data_path.resolve()),
        "input_sha256": inputs,
        "versions": {
            "python": platform.python_version(), "numpy": numpy.__version__,
            "pandas": pandas.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
