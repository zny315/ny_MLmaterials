from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


class SplitError(ValueError):
    pass


def _decode_ids(value: Any, column: str) -> tuple[str, ...]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise SplitError(f"{column} is not valid JSON") from exc
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise SplitError(f"{column} must contain a list")
    result = tuple(str(item) for item in value)
    if len(result) != len(set(result)):
        raise SplitError(f"{column} contains duplicate IDs")
    return result


@dataclass(frozen=True)
class SplitRecord:
    split_id: str
    formulation_split_id: str
    target: str
    task_type: str
    scenario: str
    axis: str
    strategy: str
    N: int
    trial: int
    train_sample_ids: tuple[str, ...]
    test_sample_ids: tuple[str, ...]


def load_splits(data_dir: str | Path, config: dict[str, Any]) -> list[SplitRecord]:
    data_dir = Path(data_dir)
    csv_path = data_dir / "split_manifest.csv"
    parquet_path = data_dir / "split_manifest.parquet"
    if csv_path.exists():
        frame = pd.read_csv(csv_path)
    elif parquet_path.exists():
        frame = pd.read_parquet(parquet_path)
    else:
        raise SplitError("missing split_manifest.csv or split_manifest.parquet")
    required = {
        "split_id", "formulation_split_id", "target", "task_type", "scenario",
        "axis", "strategy", "N", "trial", "train_sample_ids", "test_sample_ids",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise SplitError(f"split manifest missing columns: {sorted(missing)}")
    scope = config["split_scope"]
    frame = frame[
        (frame["scenario"] == scope["scenario"])
        & (frame["axis"] == scope["axis"])
        & (frame["strategy"] == scope["strategy"])
        & frame["N"].isin(scope["N"])
        & frame["target"].isin(config["targets"])
        & frame["trial"].between(scope["trials"][0], scope["trials"][1])
    ]
    if frame.empty:
        raise SplitError("no rows match split_scope")
    records: list[SplitRecord] = []
    universe: set[str] | None = None
    for row in frame.to_dict(orient="records"):
        train = _decode_ids(row["train_sample_ids"], "train_sample_ids")
        test = _decode_ids(row["test_sample_ids"], "test_sample_ids")
        if set(train).intersection(test):
            raise SplitError(f"train/test overlap in {row['split_id']}")
        if int(row["N"]) != len(train):
            raise SplitError(f"N does not match train count in {row['split_id']}")
        if universe is None:
            universe = set(train) | set(test)
        if set(train) | set(test) != universe:
            raise SplitError("all selected interpolation splits must cover one target universe")
        target = str(row["target"])
        if str(row["task_type"]) != config["targets"][target]:
            raise SplitError(f"task type mismatch for {target}")
        records.append(SplitRecord(
            split_id=str(row["split_id"]),
            formulation_split_id=str(row["formulation_split_id"]),
            target=target,
            task_type=str(row["task_type"]),
            scenario=str(row["scenario"]), axis=str(row["axis"]),
            strategy=str(row["strategy"]), N=int(row["N"]), trial=int(row["trial"]),
            train_sample_ids=train, test_sample_ids=test,
        ))
    records.sort(key=lambda item: (item.target, item.N, item.trial, item.split_id))
    return records
