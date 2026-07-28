from __future__ import annotations

from typing import Iterable

import pandas as pd

try:
    from eva_split_pipeline import compute_loading_conversions
except ModuleNotFoundError:
    from pipeline.eva_split_pipeline import compute_loading_conversions


CONVERSION_COLUMNS = (
    "entity_loading_wt_total_formulation",
    "entity_loading_phr_resin",
    "entity_loading_conversion_notes",
)


def clean_key_value(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in {"", "#", "nan", "none", "null"} else text


def loading_conversion_group_key(row: pd.Series) -> str:
    sample_uid = clean_key_value(row.get("sample_uid"))
    if sample_uid:
        return f"uid:{sample_uid}"
    values = (
        row.get("article_index"),
        row.get("sample_index_in_article"),
        row.get("sample_name"),
    )
    return "legacy:" + "||".join(clean_key_value(value) for value in values)


def changed_cell_counts(before: pd.DataFrame, after: pd.DataFrame, columns: Iterable[str]) -> dict[str, int]:
    changed = {}
    for column in columns:
        left = (
            before[column].fillna("").astype(str)
            if column in before.columns
            else pd.Series("", index=before.index, dtype=str)
        )
        right = after[column].fillna("").astype(str)
        changed[column] = int((left != right).sum())
    return changed


def normalize_entity_loading_dataframe(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    output = dataframe.copy()
    existing_values = {}
    for column in CONVERSION_COLUMNS:
        if column in output.columns:
            existing_values[column] = output[column].copy()
        else:
            output[column] = pd.NA

    group_keys = output.apply(loading_conversion_group_key, axis=1)
    for indexes in group_keys.groupby(group_keys, sort=False).groups.values():
        converted = compute_loading_conversions(output.loc[indexes])
        output.loc[indexes, converted.columns] = converted

    for column, values in existing_values.items():
        present = values.map(clean_key_value).ne("")
        output.loc[present, column] = values.loc[present]

    return output, changed_cell_counts(dataframe, output, CONVERSION_COLUMNS)
