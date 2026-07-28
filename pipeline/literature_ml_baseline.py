import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = BASE_DIR / "data" / "eva_full_pipeline_result_0606"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULT_DIR / "ml_ready_literature_baseline"

KEY_COLS = ["文章编号", "文章内样品序号", "样品名称"]
ID_COLS = ["文章编号", "文章内样品序号", "样品名称", "文章名称", "DOI", "源文件ID"]

TARGET_SPECS = {
    "LOI_rel_increase": {
        "column": "LOI",
        "kind": "regression",
        "direction": "increase",
        "min_samples": 40,
    },
    "pHRR_rel_reduction": {
        "column": "pHRR",
        "kind": "regression",
        "direction": "reduction",
        "min_samples": 40,
    },
    "THR_rel_reduction": {
        "column": "THR",
        "kind": "regression",
        "direction": "reduction",
        "min_samples": 40,
    },
    "TSP_rel_reduction": {
        "column": "TSP",
        "kind": "regression",
        "direction": "reduction",
        "min_samples": 30,
    },
    "elongation_rel_change": {
        "column": "断裂伸长率",
        "kind": "regression",
        "direction": "change",
        "min_samples": 40,
    },
    "tensile_rel_change": {
        "column": "抗拉强度",
        "kind": "regression",
        "direction": "change",
        "min_samples": 40,
    },
    "UL94_V0": {
        "column": "UL-94等级",
        "kind": "classification",
        "direction": "absolute",
        "min_samples": 40,
    },
}

PERFORMANCE_COLUMNS = [spec["column"] for spec in TARGET_SPECS.values()]
TGA_FEATURE_COLUMNS = ["T5", "T10", "T50", "Tmax1", "Tmax2", "TGA残炭率"]

RELIABILITY_NUMERIC = [
    "性能可靠度总分",
    "性能真实性评分",
    "文章可信度",
    "全局文章先验分",
]

RELIABILITY_CATEGORICAL = [
    "全局异常等级",
    "实验误差风险",
    "学术不端或异常风险",
    "推荐用途",
    "材料性能合理性",
    "配方趋势合理性",
    "横向异常性",
]

ROLE_BUCKET_PATTERNS = {
    "matrix": ["基体", "matrix", "resin", "polymer"],
    "flame_retardant": ["阻燃", "flame", "retardant", "fr"],
    "synergist": ["协效", "synerg"],
    "filler": ["填料", "filler", "reinforc"],
    "compatibilizer": ["相容", "compatibil"],
    "crosslinker": ["交联", "crosslink", "peroxide", "dcp", "taic", "tac"],
    "antioxidant": ["抗氧", "stabil", "antioxid"],
    "char_agent": ["成炭", "carbonization", "char"],
}


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"} or text in {"#", "-", "/"}:
        return ""
    return text


def normalize_key_value(value) -> str:
    text = clean_text(value)
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text


def make_key(df: pd.DataFrame) -> pd.Series:
    parts = []
    for col in KEY_COLS:
        if col in df.columns:
            parts.append(df[col].map(normalize_key_value))
        else:
            parts.append(pd.Series([""] * len(df), index=df.index))
    return parts[0] + "||" + parts[1] + "||" + parts[2]


def parse_number(value) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"} or text in {"#", "-", "/"}:
        return np.nan
    text = text.replace(",", "")
    text = re.sub(r"^[A-Za-z]+-", "", text)
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return np.nan
    try:
        return float(match.group(0))
    except ValueError:
        return np.nan


def visual_prefix(value) -> str:
    text = clean_text(value).lower()
    if text.startswith("w-"):
        return "w"
    if text.startswith("l-"):
        return "l"
    if text.startswith("v-"):
        return "v"
    return "text"


def first_nonempty(values: Iterable) -> str:
    for value in values:
        text = clean_text(value)
        if text:
            return text
    return ""


def first_numeric(values: Iterable) -> float:
    for value in values:
        number = parse_number(value)
        if not np.isnan(number):
            return number
    return np.nan


def role_bucket(role: str, entity_name: str = "") -> str:
    text = f"{role} {entity_name}".lower()
    for bucket, patterns in ROLE_BUCKET_PATTERNS.items():
        if any(pattern.lower() in text for pattern in patterns):
            return bucket
    return "other"


def safe_feature_name(prefix: str, value: str, max_len: int = 80) -> str:
    text = clean_text(value)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[\\/:*?\"<>|]+", "_", text)
    text = text.strip("_")
    if len(text) > max_len:
        text = text[:max_len].rstrip("_")
    return f"{prefix}_{text}" if text else f"{prefix}_missing"


def read_table(result_dir: Path, name: str) -> pd.DataFrame:
    path = result_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required table: {path}")
    return pd.read_excel(path)


def build_entity_features(entities: pd.DataFrame, parts: pd.DataFrame, top_entities: int) -> pd.DataFrame:
    entities = entities.copy()
    parts = parts.copy()
    entities["_key"] = make_key(entities)
    parts["_key"] = make_key(parts)

    for col in ["实体添加量_wt总配方", "实体添加量_phr树脂", "总添加量", "EVA_基体内占比", "PE_基体内占比", "POE_基体内占比", "EVA_VA含量", "熔融指数"]:
        if col in entities.columns:
            entities[f"__num_{col}"] = entities[col].map(parse_number)

    if "实体角色" not in entities.columns:
        entities["实体角色"] = ""
    if "实体名称" not in entities.columns:
        entities["实体名称"] = ""
    entities["__role_bucket"] = [
        role_bucket(role, name)
        for role, name in zip(entities["实体角色"].map(clean_text), entities["实体名称"].map(clean_text))
    ]

    rows: List[Dict] = []
    grouped = entities.groupby("_key", dropna=False, sort=False)
    for key, group in grouped:
        row: Dict = {"_key": key}
        for col in ID_COLS:
            if col in group.columns:
                row[col] = first_nonempty(group[col])

        for src, dst in [
            ("基体体系", "matrix_system"),
            ("配方计量基准", "formulation_basis"),
            ("PE类型", "pe_type"),
            ("POE类型", "poe_type"),
            ("基体", "matrix_text"),
            ("混合方式", "mixing_method"),
        ]:
            if src in group.columns:
                row[dst] = first_nonempty(group[src])

        for src, dst in [
            ("__num_EVA_基体内占比", "EVA_polymer_fraction"),
            ("__num_PE_基体内占比", "PE_polymer_fraction"),
            ("__num_POE_基体内占比", "POE_polymer_fraction"),
            ("__num_EVA_VA含量", "EVA_VA_content"),
            ("__num_熔融指数", "melt_flow_index"),
            ("__num_总添加量", "total_additive_loading"),
        ]:
            if src in group.columns:
                row[dst] = first_numeric(group[src])

        row["n_entities"] = len(group)
        row["n_non_matrix_entities"] = int((group["__role_bucket"] != "matrix").sum())
        row["n_entity_smiles"] = int(group.get("实体_SMILES", pd.Series(index=group.index, dtype=object)).map(clean_text).astype(bool).sum())
        row["has_entity_smiles"] = int(row["n_entity_smiles"] > 0)

        for bucket in sorted(set(ROLE_BUCKET_PATTERNS) | {"other"}):
            mask = group["__role_bucket"] == bucket
            row[f"role_count_{bucket}"] = int(mask.sum())
            wt = group.loc[mask, "__num_实体添加量_wt总配方"] if "__num_实体添加量_wt总配方" in group.columns else pd.Series(dtype=float)
            phr = group.loc[mask, "__num_实体添加量_phr树脂"] if "__num_实体添加量_phr树脂" in group.columns else pd.Series(dtype=float)
            row[f"role_wt_sum_{bucket}"] = float(np.nansum(wt)) if len(wt) else 0.0
            row[f"role_phr_sum_{bucket}"] = float(np.nansum(phr)) if len(phr) else 0.0

        for src, prefix, top_values in [
            ("实体材料类型", "entity_material", None),
            ("实体类型", "entity_type", None),
            ("实体名称", "entity_name", top_entities),
        ]:
            if src not in group.columns:
                continue
            values = group[src].map(clean_text)
            if top_values:
                # Filled later after global top list is known.
                continue
            for value, count in values[values.astype(bool)].value_counts().items():
                row[safe_feature_name(prefix, value)] = int(count)
        rows.append(row)

    features = pd.DataFrame(rows)

    if "实体名称" in entities.columns and top_entities > 0:
        top_names = (
            entities["实体名称"].map(clean_text)
            .loc[lambda s: s.astype(bool)]
            .value_counts()
            .head(top_entities)
            .index
            .tolist()
        )
        entity_counts = (
            entities[entities["实体名称"].map(clean_text).isin(top_names)]
            .assign(__entity_name=lambda d: d["实体名称"].map(clean_text))
            .groupby(["_key", "__entity_name"])
            .size()
            .unstack(fill_value=0)
        )
        entity_counts.columns = [safe_feature_name("entity_name", col) for col in entity_counts.columns]
        features = features.merge(entity_counts.reset_index(), on="_key", how="left")

    if len(parts):
        part_rows = []
        for key, group in parts.groupby("_key", dropna=False, sort=False):
            row = {"_key": key}
            row["n_parts"] = len(group)
            row["n_part_smiles"] = int(group.get("部件_SMILES", pd.Series(index=group.index, dtype=object)).map(clean_text).astype(bool).sum())
            row["has_part_smiles"] = int(row["n_part_smiles"] > 0)
            for src, prefix in [("部件角色", "part_role"), ("部件关系", "part_relation"), ("改性方式", "modification_mode")]:
                if src not in group.columns:
                    continue
                for value, count in group[src].map(clean_text).loc[lambda s: s.astype(bool)].value_counts().items():
                    row[safe_feature_name(prefix, value)] = int(count)
            part_rows.append(row)
        part_features = pd.DataFrame(part_rows)
        features = features.merge(part_features, on="_key", how="left")

    features = features.fillna({col: 0 for col in features.columns if col.startswith(("role_", "entity_", "part_", "modification_", "n_", "has_"))})
    return features


def build_performance_features(performance: pd.DataFrame) -> pd.DataFrame:
    perf = performance.copy()
    perf["_key"] = make_key(perf)

    rows = []
    for key, group in perf.groupby("_key", dropna=False, sort=False):
        row = {"_key": key}
        for col in ID_COLS:
            if col in group.columns:
                row[col] = first_nonempty(group[col])

        for col in PERFORMANCE_COLUMNS + TGA_FEATURE_COLUMNS:
            if col in group.columns:
                value = first_nonempty(group[col])
                row[f"raw_{col}"] = value
                row[f"value_{col}"] = parse_number(value)
                row[f"source_{col}"] = visual_prefix(value) if value else ""

        for col in RELIABILITY_NUMERIC:
            if col in group.columns:
                row[f"reliability_{col}"] = first_numeric(group[col])
        for col in RELIABILITY_CATEGORICAL:
            if col in group.columns:
                row[f"reliability_{col}"] = first_nonempty(group[col])
        rows.append(row)
    return pd.DataFrame(rows)


def is_eva_sample(row: pd.Series) -> bool:
    checks = [
        clean_text(row.get("matrix_system", "")),
        clean_text(row.get("matrix_text", "")),
    ]
    if any("eva" in text.lower() for text in checks):
        return True
    value = row.get("EVA_polymer_fraction", np.nan)
    return not pd.isna(value) and float(value) > 0


def matrix_signature(row: pd.Series) -> str:
    system = clean_text(row.get("matrix_system", "")).lower()
    if not system:
        system = "eva"
    eva = parse_number(row.get("EVA_polymer_fraction", np.nan))
    pe = parse_number(row.get("PE_polymer_fraction", np.nan))
    poe = parse_number(row.get("POE_polymer_fraction", np.nan))
    parts = [system]
    for label, value in [("eva", eva), ("pe", pe), ("poe", poe)]:
        if not np.isnan(value):
            parts.append(f"{label}{round(value / 5) * 5:.0f}")
    return "|".join(parts)


def control_hint(sample_name: str) -> int:
    text = clean_text(sample_name).lower()
    if not text:
        return 0
    if re.search(r"\b(neat|pure|blank|control|reference|virgin|base)\b", text):
        return 1
    if re.fullmatch(r"eva[-_ ]?0?", text) or re.search(r"eva[-_/ ]?0\b", text):
        return 1
    if re.search(r"\b0\s*(phr|wt|%)\b", text):
        return 1
    return 0


def add_control_and_targets(sample_df: pd.DataFrame, max_control_rank_score: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = sample_df.copy()
    df["matrix_signature"] = df.apply(matrix_signature, axis=1)
    df["control_hint"] = df["样品名称"].map(control_hint) if "样品名称" in df.columns else 0

    fr_related = [
        "role_wt_sum_flame_retardant",
        "role_wt_sum_synergist",
        "role_wt_sum_filler",
        "role_phr_sum_flame_retardant",
        "role_phr_sum_synergist",
        "role_phr_sum_filler",
    ]
    for col in fr_related:
        if col not in df.columns:
            df[col] = 0.0
    df["fr_related_loading_score"] = df[fr_related].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    df["control_rank_score"] = (
        df["fr_related_loading_score"].fillna(0) * 10
        + pd.to_numeric(df.get("total_additive_loading", 0), errors="coerce").fillna(0)
        - df["control_hint"].fillna(0) * 20
    )

    control_map: Dict[Tuple[str, str], str] = {}
    control_rows = []
    for (article_id, signature), group in df.groupby(["文章编号", "matrix_signature"], dropna=False, sort=False):
        idx = group["control_rank_score"].astype(float).idxmin()
        control_key = df.loc[idx, "_key"]
        control_map[(normalize_key_value(article_id), signature)] = control_key
        control_rows.append(
            {
                "文章编号": article_id,
                "matrix_signature": signature,
                "control_key": control_key,
                "control_sample_name": df.loc[idx, "样品名称"],
                "control_rank_score": df.loc[idx, "control_rank_score"],
            }
        )

    controls = pd.DataFrame(control_rows)
    controls["control_is_usable"] = (
        pd.to_numeric(controls["control_rank_score"], errors="coerce") <= max_control_rank_score
    ).astype(int)
    usable_control_map = {
        (normalize_key_value(row["文章编号"]), row["matrix_signature"]): int(row["control_is_usable"])
        for _, row in controls.iterrows()
    }

    df["control_key"] = [
        control_map.get((normalize_key_value(article), signature), "")
        for article, signature in zip(df["文章编号"], df["matrix_signature"])
    ]
    df["control_is_usable"] = [
        usable_control_map.get((normalize_key_value(article), signature), 0)
        for article, signature in zip(df["文章编号"], df["matrix_signature"])
    ]
    control_lookup = df.set_index("_key")

    target_rows = []
    for _, row in df.iterrows():
        target_row = {
            "_key": row["_key"],
            "文章编号": row.get("文章编号", ""),
            "文章内样品序号": row.get("文章内样品序号", ""),
            "样品名称": row.get("样品名称", ""),
            "文章名称": row.get("文章名称", ""),
            "DOI": row.get("DOI", ""),
            "matrix_signature": row.get("matrix_signature", ""),
            "control_key": row.get("control_key", ""),
            "control_is_usable": int(row.get("control_is_usable", 0)),
        }
        control_key = row.get("control_key", "")
        control = control_lookup.loc[control_key] if control_key in control_lookup.index else None
        target_row["control_sample_name"] = control.get("样品名称", "") if control is not None else ""
        target_row["is_control_sample"] = int(control_key == row["_key"])

        for target_name, spec in TARGET_SPECS.items():
            col = spec["column"]
            sample_value = parse_number(row.get(f"value_{col}", np.nan))
            target_row[f"{target_name}_sample_value"] = sample_value
            target_row[f"{target_name}_source"] = row.get(f"source_{col}", "")
            if spec["kind"] == "classification":
                if np.isnan(sample_value):
                    target_row[target_name] = np.nan
                else:
                    target_row[target_name] = int(sample_value == 0)
                continue
            control_value = parse_number(control.get(f"value_{col}", np.nan)) if control is not None else np.nan
            target_row[f"{target_name}_control_value"] = control_value
            if not target_row["control_is_usable"] or np.isnan(sample_value) or np.isnan(control_value) or control_value == 0:
                target_row[target_name] = np.nan
            elif spec["direction"] == "reduction":
                target_row[target_name] = (control_value - sample_value) / control_value
            else:
                target_row[target_name] = (sample_value - control_value) / control_value
        target_rows.append(target_row)

    return df, pd.DataFrame(target_rows), controls


def one_hot_frame(df: pd.DataFrame, columns: List[str], prefix: str) -> pd.DataFrame:
    frames = []
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].map(clean_text)
        if not series.astype(bool).any():
            continue
        dummies = pd.get_dummies(series, prefix=f"{prefix}_{col}", dummy_na=False)
        frames.append(dummies)
    if not frames:
        return pd.DataFrame(index=df.index)
    return pd.concat(frames, axis=1)


def make_model_matrix(samples: pd.DataFrame, include_tga_features: bool) -> Tuple[pd.DataFrame, List[str], List[str]]:
    exclude_prefixes = ("raw_", "value_", "source_")
    numeric_cols = []
    reliability_cols = []
    for col in samples.columns:
        if col in {"_key"} or col in ID_COLS:
            continue
        if col.startswith("value_"):
            source_col = col.replace("value_", "", 1)
            if include_tga_features and source_col in TGA_FEATURE_COLUMNS:
                numeric_cols.append(col)
            continue
        if col.startswith(("raw_", "source_")):
            continue
        if col in {"control_key", "matrix_signature", "control_hint"}:
            continue
        if col.startswith("reliability_"):
            if col.replace("reliability_", "") in RELIABILITY_NUMERIC:
                reliability_cols.append(col)
            continue
        if not include_tga_features and col in [f"value_{c}" for c in TGA_FEATURE_COLUMNS]:
            continue
        if pd.api.types.is_numeric_dtype(samples[col]):
            numeric_cols.append(col)

    base_numeric = samples[numeric_cols].apply(pd.to_numeric, errors="coerce")
    categorical_cols = ["matrix_system", "formulation_basis", "pe_type", "poe_type", "mixing_method"]
    categorical = one_hot_frame(samples, categorical_cols, "cat")
    reliability_cat = one_hot_frame(samples, [f"reliability_{col}" for col in RELIABILITY_CATEGORICAL], "meta")
    reliability_num = samples[reliability_cols].apply(pd.to_numeric, errors="coerce") if reliability_cols else pd.DataFrame(index=samples.index)

    formulation = pd.concat([base_numeric, categorical], axis=1)
    reliability = pd.concat([reliability_num, reliability_cat], axis=1)
    matrix = pd.concat([formulation, reliability], axis=1)
    matrix = matrix.loc[:, ~matrix.columns.duplicated()]
    return matrix, list(formulation.columns), list(reliability.columns)


def import_sklearn():
    try:
        from sklearn.compose import TransformedTargetRegressor  # noqa: F401
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            f1_score,
            mean_absolute_error,
            mean_squared_error,
            r2_score,
            roc_auc_score,
        )
        from sklearn.model_selection import GroupKFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return {
            "RandomForestClassifier": RandomForestClassifier,
            "RandomForestRegressor": RandomForestRegressor,
            "SimpleImputer": SimpleImputer,
            "LogisticRegression": LogisticRegression,
            "Ridge": Ridge,
            "accuracy_score": accuracy_score,
            "balanced_accuracy_score": balanced_accuracy_score,
            "f1_score": f1_score,
            "mean_absolute_error": mean_absolute_error,
            "mean_squared_error": mean_squared_error,
            "r2_score": r2_score,
            "roc_auc_score": roc_auc_score,
            "GroupKFold": GroupKFold,
            "Pipeline": Pipeline,
            "StandardScaler": StandardScaler,
        }
    except Exception:
        return None


def sample_weights(rows: pd.DataFrame) -> np.ndarray:
    scores = pd.to_numeric(rows.get("reliability_性能可靠度总分", pd.Series(index=rows.index, dtype=float)), errors="coerce")
    weights = (scores / 100.0).clip(lower=0.25, upper=1.0).fillna(0.7)
    usage = rows.get("reliability_推荐用途", pd.Series([""] * len(rows), index=rows.index)).map(clean_text)
    weights = weights.where(~usage.eq("exclude"), 0.1)
    weights = weights.where(~usage.eq("manual_check"), weights * 0.6)
    weights = weights.where(~usage.eq("use_with_caution"), weights * 0.8)
    return weights.to_numpy(dtype=float)


def run_model_baselines(samples: pd.DataFrame, targets: pd.DataFrame, feature_matrix: pd.DataFrame, formulation_cols: List[str], reliability_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sk = import_sklearn()
    if sk is None:
        return pd.DataFrame([{"status": "sklearn_missing", "message": "Install scikit-learn to run model baselines."}]), pd.DataFrame()

    merged = samples.merge(targets, on=["_key"] + [col for col in ID_COLS if col in targets.columns and col in samples.columns], how="left", suffixes=("", "_target"))
    groups = merged["文章编号"].map(normalize_key_value)
    metrics_rows = []
    pred_rows = []

    feature_sets = {
        "formulation_only": formulation_cols,
        "formulation_plus_reliability_meta": formulation_cols + reliability_cols,
        "formulation_weighted_by_reliability": formulation_cols,
    }

    for target_name, spec in TARGET_SPECS.items():
        if target_name not in merged.columns:
            continue
        y_all = pd.to_numeric(merged[target_name], errors="coerce")
        valid_mask = y_all.notna() & groups.notna()
        if spec["kind"] == "regression":
            valid_mask &= np.isfinite(y_all)
        data = merged.loc[valid_mask].copy()
        y = y_all.loc[valid_mask].to_numpy()
        target_groups = groups.loc[valid_mask].to_numpy()
        n_articles = len(pd.unique(target_groups))
        n_samples = len(data)
        if n_samples < spec["min_samples"] or n_articles < 3:
            metrics_rows.append(
                {
                    "target": target_name,
                    "model": "skipped",
                    "feature_set": "",
                    "n_samples": n_samples,
                    "n_articles": n_articles,
                    "status": "too_few_samples_or_articles",
                }
            )
            continue
        if spec["kind"] == "classification" and len(np.unique(y)) < 2:
            metrics_rows.append(
                {
                    "target": target_name,
                    "model": "skipped",
                    "feature_set": "",
                    "n_samples": n_samples,
                    "n_articles": n_articles,
                    "status": "single_class",
                }
            )
            continue

        n_splits = min(5, n_articles)
        splitter = sk["GroupKFold"](n_splits=n_splits)

        for feature_set_name, cols in feature_sets.items():
            cols = [col for col in cols if col in feature_matrix.columns]
            X = feature_matrix.loc[valid_mask, cols]
            if X.shape[1] == 0:
                continue

            if spec["kind"] == "regression":
                model_defs = {
                    "ridge": sk["Pipeline"](
                        [
                            ("imputer", sk["SimpleImputer"](strategy="median")),
                            ("scaler", sk["StandardScaler"]()),
                            ("model", sk["Ridge"](alpha=3.0)),
                        ]
                    ),
                    "random_forest": sk["Pipeline"](
                        [
                            ("imputer", sk["SimpleImputer"](strategy="median")),
                            (
                                "model",
                                sk["RandomForestRegressor"](
                                    n_estimators=160,
                                    random_state=42,
                                    min_samples_leaf=3,
                                    n_jobs=-1,
                                ),
                            ),
                        ]
                    ),
                }
            else:
                model_defs = {
                    "logistic": sk["Pipeline"](
                        [
                            ("imputer", sk["SimpleImputer"](strategy="median")),
                            ("scaler", sk["StandardScaler"]()),
                            ("model", sk["LogisticRegression"](max_iter=2000, class_weight="balanced")),
                        ]
                    ),
                    "random_forest": sk["Pipeline"](
                        [
                            ("imputer", sk["SimpleImputer"](strategy="median")),
                            (
                                "model",
                                sk["RandomForestClassifier"](
                                    n_estimators=160,
                                    random_state=42,
                                    min_samples_leaf=3,
                                    class_weight="balanced",
                                    n_jobs=-1,
                                ),
                            ),
                        ]
                    ),
                }

            for model_name, model in model_defs.items():
                y_pred = np.full_like(y, np.nan, dtype=float)
                y_score = np.full_like(y, np.nan, dtype=float)
                baseline_pred = np.full_like(y, np.nan, dtype=float)
                for train_idx, test_idx in splitter.split(X, y, groups=target_groups):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train = y[train_idx]
                    fit_kwargs = {}
                    if feature_set_name == "formulation_weighted_by_reliability":
                        weights = sample_weights(data.iloc[train_idx])
                        final_step = "model"
                        fit_kwargs[f"{final_step}__sample_weight"] = weights
                    model.fit(X_train, y_train, **fit_kwargs)
                    if spec["kind"] == "classification":
                        pred = model.predict(X_test)
                        y_pred[test_idx] = pred.astype(float)
                        if hasattr(model[-1], "predict_proba"):
                            y_score[test_idx] = model.predict_proba(X_test)[:, 1]
                        baseline_value = int(np.nanmean(y_train) >= 0.5)
                        baseline_pred[test_idx] = baseline_value
                    else:
                        y_pred[test_idx] = model.predict(X_test)
                        baseline_pred[test_idx] = np.nanmean(y_train)

                if spec["kind"] == "classification":
                    metrics = {
                        "accuracy": sk["accuracy_score"](y, y_pred),
                        "balanced_accuracy": sk["balanced_accuracy_score"](y, y_pred),
                        "f1": sk["f1_score"](y, y_pred, zero_division=0),
                        "baseline_accuracy": sk["accuracy_score"](y, baseline_pred),
                    }
                    if len(np.unique(y)) == 2 and np.isfinite(y_score).all():
                        try:
                            metrics["roc_auc"] = sk["roc_auc_score"](y, y_score)
                        except Exception:
                            metrics["roc_auc"] = np.nan
                else:
                    rmse = math.sqrt(sk["mean_squared_error"](y, y_pred))
                    baseline_rmse = math.sqrt(sk["mean_squared_error"](y, baseline_pred))
                    metrics = {
                        "mae": sk["mean_absolute_error"](y, y_pred),
                        "rmse": rmse,
                        "r2": sk["r2_score"](y, y_pred),
                        "baseline_mae": sk["mean_absolute_error"](y, baseline_pred),
                        "baseline_rmse": baseline_rmse,
                    }

                metrics_rows.append(
                    {
                        "target": target_name,
                        "source_column": spec["column"],
                        "kind": spec["kind"],
                        "model": model_name,
                        "feature_set": feature_set_name,
                        "n_samples": n_samples,
                        "n_articles": n_articles,
                        "n_splits": n_splits,
                        "n_features": len(cols),
                        "status": "success",
                        **metrics,
                    }
                )

                for i, (_, data_row) in enumerate(data.iterrows()):
                    pred_rows.append(
                        {
                            "target": target_name,
                            "model": model_name,
                            "feature_set": feature_set_name,
                            "_key": data_row["_key"],
                            "文章编号": data_row.get("文章编号", ""),
                            "文章内样品序号": data_row.get("文章内样品序号", ""),
                            "样品名称": data_row.get("样品名称", ""),
                            "y_true": y[i],
                            "y_pred": y_pred[i],
                            "y_score": y_score[i] if spec["kind"] == "classification" else np.nan,
                        }
                    )

    return pd.DataFrame(metrics_rows), pd.DataFrame(pred_rows)


def feature_coverage(samples: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target_name, spec in TARGET_SPECS.items():
        if target_name in targets.columns:
            nonempty = pd.to_numeric(targets[target_name], errors="coerce").notna()
            rows.append(
                {
                    "target": target_name,
                    "source_column": spec["column"],
                    "kind": spec["kind"],
                    "nonempty_samples": int(nonempty.sum()),
                    "nonempty_articles": int(targets.loc[nonempty, "文章编号"].map(normalize_key_value).nunique()),
                }
            )

    for col in samples.columns:
        if col.startswith("value_") or col.startswith("reliability_") or col in ["EVA_polymer_fraction", "PE_polymer_fraction", "POE_polymer_fraction", "EVA_VA_content"]:
            rows.append(
                {
                    "target": f"feature::{col}",
                    "source_column": col,
                    "kind": "feature",
                    "nonempty_samples": int(samples[col].notna().sum()),
                    "nonempty_articles": int(samples.loc[samples[col].notna(), "文章编号"].map(normalize_key_value).nunique()) if "文章编号" in samples.columns else np.nan,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ML-ready literature baseline tables from EVA extraction outputs.")
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR), help="Directory containing EVA output xlsx files.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Default: result-dir/ml_ready_literature_baseline.")
    parser.add_argument("--top-entities", type=int, default=120, help="Top entity names to one-hot count. Default: 120.")
    parser.add_argument("--include-tga-features", action="store_true", help="Include TGA performance fields as input features for non-TGA targets.")
    parser.add_argument("--include-non-eva", action="store_true", help="Do not filter to samples with EVA matrix evidence.")
    parser.add_argument("--max-control-rank-score", type=float, default=20.0, help="Only use automatically selected controls with rank score at or below this value for relative targets. Default: 20.")
    parser.add_argument("--skip-models", action="store_true", help="Only build ML-ready tables; skip scikit-learn baselines.")
    args = parser.parse_args()

    result_dir = Path(args.result_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else result_dir / "ml_ready_literature_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    entities = read_table(result_dir, "formulation_entities.xlsx")
    parts = read_table(result_dir, "formulation_entity_parts.xlsx")
    performance = read_table(result_dir, "performance_samples_final.xlsx")

    entity_features = build_entity_features(entities, parts, args.top_entities)
    performance_features = build_performance_features(performance)
    samples = entity_features.merge(performance_features, on="_key", how="inner", suffixes=("", "_perf"))
    for col in ID_COLS:
        perf_col = f"{col}_perf"
        if perf_col in samples.columns:
            samples[col] = samples[col].where(samples[col].map(clean_text).astype(bool), samples[perf_col])
            samples.drop(columns=[perf_col], inplace=True)

    samples["is_eva_sample"] = samples.apply(is_eva_sample, axis=1).astype(int)
    if not args.include_non_eva:
        samples = samples[samples["is_eva_sample"] == 1].copy()

    samples, targets, controls = add_control_and_targets(samples, args.max_control_rank_score)
    samples = samples.reset_index(drop=True)
    targets = targets.reset_index(drop=True)
    feature_matrix, formulation_cols, reliability_cols = make_model_matrix(samples, args.include_tga_features)

    metrics = pd.DataFrame()
    predictions = pd.DataFrame()
    if not args.skip_models:
        metrics, predictions = run_model_baselines(samples, targets, feature_matrix, formulation_cols, reliability_cols)

    coverage = feature_coverage(samples, targets)
    summary = {
        "result_dir": str(result_dir),
        "output_dir": str(output_dir),
        "n_entity_rows": int(len(entities)),
        "n_part_rows": int(len(parts)),
        "n_performance_rows": int(len(performance)),
        "n_ml_samples": int(len(samples)),
        "n_articles": int(samples["文章编号"].map(normalize_key_value).nunique()) if "文章编号" in samples.columns else 0,
        "eva_only": not args.include_non_eva,
        "include_tga_features": bool(args.include_tga_features),
        "n_formulation_features": len(formulation_cols),
        "n_reliability_features": len(reliability_cols),
        "max_control_rank_score": args.max_control_rank_score,
        "n_control_groups": int(len(controls)),
        "n_usable_control_groups": int(controls["control_is_usable"].sum()) if "control_is_usable" in controls.columns else 0,
    }

    samples_path = output_dir / "sample_feature_table.xlsx"
    targets_path = output_dir / "target_table.xlsx"
    controls_path = output_dir / "control_sample_map.xlsx"
    metrics_path = output_dir / "baseline_metrics.xlsx"
    predictions_path = output_dir / "baseline_predictions.xlsx"
    coverage_path = output_dir / "coverage_summary.xlsx"
    matrix_path = output_dir / "model_matrix.csv"
    summary_path = output_dir / "run_summary.json"

    samples.to_excel(samples_path, index=False)
    targets.to_excel(targets_path, index=False)
    controls.to_excel(controls_path, index=False)
    coverage.to_excel(coverage_path, index=False)
    feature_matrix.insert(0, "_key", samples["_key"].to_numpy())
    feature_matrix.to_csv(matrix_path, index=False, encoding="utf-8-sig")
    if len(metrics):
        metrics.to_excel(metrics_path, index=False)
    if len(predictions):
        predictions.to_excel(predictions_path, index=False)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
