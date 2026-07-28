from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


class DataContractError(ValueError):
    pass


def _read_table(directory: Path, stem: str) -> pd.DataFrame:
    csv_path = directory / f"{stem}.csv"
    parquet_path = directory / f"{stem}.parquet"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except ImportError as exc:
            raise DataContractError(
                f"reading {parquet_path.name} requires the optional parquet dependencies"
            ) from exc
    raise DataContractError(f"missing {csv_path.name} or {parquet_path.name}")


@dataclass(frozen=True)
class Snapshot:
    root: Path
    X_source_full: np.ndarray
    X_source_compact: np.ndarray
    X_target_full: np.ndarray
    X_target_compact: np.ndarray
    source_ids: np.ndarray
    target_ids: np.ndarray
    source_reliability_eligible: np.ndarray
    source_targets: pd.DataFrame
    target_samples: pd.DataFrame

    @classmethod
    def load(cls, root: str | Path, targets: dict[str, str]) -> "Snapshot":
        root = Path(root)
        feature_path = root / "features.npz"
        if not feature_path.exists():
            raise DataContractError(f"missing {feature_path}")
        arrays = np.load(feature_path, allow_pickle=False)
        required = {
            "X_source_full", "X_source_compact", "X_target_full",
            "X_target_compact", "source_sample_ids", "target_sample_ids",
            "source_reliability_eligible",
        }
        missing = required.difference(arrays.files)
        if missing:
            raise DataContractError(f"features.npz missing arrays: {sorted(missing)}")
        source_ids = arrays["source_sample_ids"].astype(str)
        target_ids = arrays["target_sample_ids"].astype(str)
        source_targets = _read_table(root, "source_targets")
        target_samples = _read_table(root, "target_samples")
        # The original prepared snapshot called the literature identifier
        # ``_key``.  Public snapshots may use the clearer ``sample_id`` name.
        if "sample_id" not in source_targets and "_key" in source_targets:
            source_targets = source_targets.rename(columns={"_key": "sample_id"})
        for label, frame, ids in (
            ("source_targets", source_targets, source_ids),
            ("target_samples", target_samples, target_ids),
        ):
            if "sample_id" not in frame:
                raise DataContractError(f"{label} missing sample_id")
            frame["sample_id"] = frame["sample_id"].astype(str)
            if frame["sample_id"].duplicated().any():
                raise DataContractError(f"{label} contains duplicate sample_id")
            if set(frame["sample_id"]) != set(ids):
                raise DataContractError(f"{label} sample IDs do not match features.npz")
        source_targets = source_targets.set_index("sample_id").loc[source_ids].reset_index()
        target_samples = target_samples.set_index("sample_id").loc[target_ids].reset_index()
        for target in targets:
            if target not in source_targets or target not in target_samples:
                raise DataContractError(f"target column missing from snapshot: {target}")
        matrices = {
            name: np.asarray(arrays[name], dtype=np.float64)
            for name in ("X_source_full", "X_source_compact", "X_target_full", "X_target_compact")
        }
        for name, matrix in matrices.items():
            expected = len(source_ids) if "source" in name else len(target_ids)
            if matrix.ndim != 2 or len(matrix) != expected or not np.isfinite(matrix).all():
                raise DataContractError(f"{name} must be a finite {expected}-row matrix")
        eligible = np.asarray(arrays["source_reliability_eligible"], dtype=bool)
        if eligible.shape != (len(source_ids),):
            raise DataContractError("source_reliability_eligible has wrong shape")
        if not eligible.any():
            raise DataContractError("no literature rows are reliability eligible")
        return cls(
            root=root,
            source_ids=source_ids,
            target_ids=target_ids,
            source_reliability_eligible=eligible,
            source_targets=source_targets,
            target_samples=target_samples,
            **matrices,
        )

    @property
    def target_position(self) -> dict[str, int]:
        return {sample_id: index for index, sample_id in enumerate(self.target_ids)}

    def positions(self, sample_ids: tuple[str, ...] | list[str]) -> np.ndarray:
        mapping = self.target_position
        try:
            return np.asarray([mapping[str(value)] for value in sample_ids], dtype=int)
        except KeyError as exc:
            raise DataContractError(f"unknown target sample ID: {exc.args[0]}") from exc

    def target_y(self, target: str) -> np.ndarray:
        values = pd.to_numeric(self.target_samples[target], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise DataContractError(f"target outcomes contain missing/nonfinite values: {target}")
        return values

    def source_endpoint(self, target: str, *, view: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = pd.to_numeric(self.source_targets[target], errors="coerce").to_numpy(float)
        keep = self.source_reliability_eligible & np.isfinite(values)
        if not keep.any():
            raise DataContractError(f"no eligible literature outcomes for {target}")
        matrix = self.X_source_full if view == "full" else self.X_source_compact
        return matrix[keep], values[keep], self.source_ids[keep]

    def target_X(self, *, view: str) -> np.ndarray:
        if view == "full":
            return self.X_target_full
        if view == "compact":
            return self.X_target_compact
        raise DataContractError(f"unknown feature view: {view}")
