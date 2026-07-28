import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from public_utils import console_print as print, install_ascii_exception_hook
    from schema import canonicalize_dataframe_columns
except ModuleNotFoundError:
    from pipeline.public_utils import console_print as print, install_ascii_exception_hook
    from pipeline.schema import canonicalize_dataframe_columns

install_ascii_exception_hook()


DEFAULT_TABLES = (
    "formulation_entities.xlsx",
    "formulation_entity_parts.xlsx",
)
MISSING = {"", "#", "nan", "none", "null", "n/a", "na"}
NUM_RE = re.compile(r"([wl]-)?([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")


def is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip().lower() in MISSING


def normalize_exact_dedupe_value(value) -> str:
    if is_missing(value):
        return ""
    text = " ".join(str(value).strip().split())
    match = NUM_RE.fullmatch(text)
    if not match:
        return text
    prefix = match.group(1) or ""
    try:
        number = float(match.group(2))
    except ValueError:
        return text
    if number.is_integer():
        return f"{prefix}{int(number)}"
    return f"{prefix}{number:.12g}"


def exact_duplicate_mask(df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.Series:
    use_columns = [col for col in (columns or df.columns) if col in df.columns]
    if not use_columns:
        return pd.Series(False, index=df.index)
    normalized = df[use_columns].apply(lambda col: col.map(normalize_exact_dedupe_value))
    return normalized.duplicated(keep="first")


def normalize_file(path: Path, apply: bool) -> dict:
    df = canonicalize_dataframe_columns(
        pd.read_excel(path, dtype=str, keep_default_na=False).fillna(""),
        context=str(path),
    )
    duplicate_mask = exact_duplicate_mask(df)
    duplicate_rows = df.loc[duplicate_mask].copy()
    result = {
        "file": str(path),
        "rows_before": int(len(df)),
        "duplicate_rows": int(duplicate_mask.sum()),
        "rows_after": int(len(df) - duplicate_mask.sum()),
        "applied": bool(apply),
    }
    if apply and duplicate_mask.any():
        df.loc[~duplicate_mask].to_excel(path, index=False)
    return result, duplicate_rows


def main():
    parser = argparse.ArgumentParser(
        description="Dry-run or apply exact duplicate-row normalization for EVA pipeline entity tables."
    )
    parser.add_argument("--output-dir", required=True, help="Pipeline output directory containing formulation/entity-part xlsx files.")
    parser.add_argument(
        "--tables",
        nargs="*",
        default=list(DEFAULT_TABLES),
        help="Workbook filenames to normalize. Default: formulation_entities.xlsx formulation_entity_parts.xlsx",
    )
    parser.add_argument("--apply", action="store_true", help="Write de-duplicated workbooks in place. Omit for dry-run.")
    parser.add_argument("--report-dir", default=None, help="Directory for JSON and duplicate-row CSV reports.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir) if args.report_dir else output_dir / "normalization_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for table in args.tables:
        path = output_dir / table
        if not path.exists():
            summary.append({"file": str(path), "error": "missing", "applied": bool(args.apply)})
            continue
        result, duplicate_rows = normalize_file(path, apply=args.apply)
        summary.append(result)
        if not duplicate_rows.empty:
            report_name = path.stem + "_exact_duplicate_rows.csv"
            duplicate_rows.to_csv(report_dir / report_name, index=False, encoding="utf-8-sig")

    (report_dir / "exact_duplicate_normalization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"report_dir={report_dir}")
    if not args.apply:
        print("dry_run=true; no workbook was modified")


if __name__ == "__main__":
    main()
