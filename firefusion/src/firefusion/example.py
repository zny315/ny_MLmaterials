from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = {
    "LOI_rel_increase": "reg",
    "pHRR_rel_reduction": "reg",
    "UL94_V0": "clf",
}


def _digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def generate_example(output: str | Path, *, seed: int = 7) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_source, n_target, p_full, p_compact = 80, 12, 8, 5
    X_source_full = rng.normal(size=(n_source, p_full))
    X_target_full = rng.normal(loc=0.25, scale=1.1, size=(n_target, p_full))
    X_source_compact = X_source_full[:, :p_compact]
    X_target_compact = X_target_full[:, :p_compact]
    source_ids = np.asarray([f"lit-{i:03d}" for i in range(n_source)])
    target_ids = np.asarray([f"lab-{i:03d}" for i in range(n_target)])
    source_loi = 0.6 * X_source_full[:, 0] - 0.35 * X_source_full[:, 1] + rng.normal(0, 0.35, n_source)
    target_loi = 0.2 + 0.75 * X_target_full[:, 0] - 0.3 * X_target_full[:, 1] + rng.normal(0, 0.3, n_target)
    source_phrr = -0.5 * X_source_full[:, 2] + 0.45 * X_source_full[:, 3] + rng.normal(0, 0.4, n_source)
    target_phrr = -0.1 - 0.65 * X_target_full[:, 2] + 0.5 * X_target_full[:, 3] + rng.normal(0, 0.35, n_target)
    source_ul94 = (X_source_full[:, 0] + 0.7 * X_source_full[:, 4] + rng.normal(0, 0.8, n_source) > 0).astype(int)
    target_ul94 = (X_target_full[:, 0] + 0.7 * X_target_full[:, 4] > np.median(X_target_full[:, 0] + 0.7 * X_target_full[:, 4])).astype(int)
    eligible = np.ones(n_source, dtype=bool)
    eligible[[3, 17]] = False
    np.savez(
        output / "features.npz",
        X_source_full=X_source_full, X_source_compact=X_source_compact,
        X_target_full=X_target_full, X_target_compact=X_target_compact,
        source_sample_ids=source_ids, target_sample_ids=target_ids,
        source_reliability_eligible=eligible,
    )
    pd.DataFrame({
        "sample_id": source_ids, "LOI_rel_increase": source_loi,
        "pHRR_rel_reduction": source_phrr, "UL94_V0": source_ul94,
    }).to_csv(output / "source_targets.csv", index=False)
    pd.DataFrame({
        "sample_id": target_ids, "LOI_rel_increase": target_loi,
        "pHRR_rel_reduction": target_phrr, "UL94_V0": target_ul94,
    }).to_csv(output / "target_samples.csv", index=False)
    rows = []
    for trial in range(2):
        train_position = rng.choice(n_target, size=3, replace=False)
        train = tuple(target_ids[train_position].tolist())
        test = tuple(value for value in target_ids.tolist() if value not in set(train))
        formulation_payload = {
            "scenario": "interp_random", "axis": "N", "strategy": "random",
            "N": 3, "trial": trial, "train": train, "test": test,
        }
        formulation_id = _digest(formulation_payload)
        for target, task in TARGETS.items():
            rows.append({
                "split_id": _digest({"formulation_split_id": formulation_id, "target": target, "task": task}),
                "formulation_split_id": formulation_id, "target": target, "task_type": task,
                "scenario": "interp_random", "axis": "N", "strategy": "random",
                "N": 3, "trial": trial,
                "train_sample_ids": json.dumps(train), "test_sample_ids": json.dumps(test),
            })
    pd.DataFrame(rows).to_csv(output / "split_manifest.csv", index=False)
    return output
