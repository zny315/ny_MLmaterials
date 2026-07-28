import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl import load_workbook

try:
    from public_utils import console_print as print, install_ascii_exception_hook
except ModuleNotFoundError:
    from pipeline.public_utils import console_print as print, install_ascii_exception_hook

install_ascii_exception_hook()


MOJIBAKE_RE = re.compile(
    "|".join(
        [
            "\u00c2",
            "\u00c3",
            "\u00e2",
            "\ufffd",
        ]
    )
)

LOCAL_MOJIBAKE_REPLACEMENTS = {
    "\u00c2\u00b0": "\u00b0",
    "\u00c2\u00b1": "\u00b1",
    "\u00c2\u00b5": "\u00b5",
    "\u00c2\u00b7": "\u00b7",
    "\u00c2\u00ae": "\u00ae",
    "\u00c3\u0097": "\u00d7",
    "\u00e2\u0080\u0090": "\u2010",
    "\u00e2\u0080\u0091": "\u2011",
    "\u00e2\u0080\u0092": "\u2012",
    "\u00e2\u0080\u0093": "\u2013",
    "\u00e2\u0080\u0094": "\u2014",
    "\u00e2\u0080\u0098": "\u2018",
    "\u00e2\u0080\u0099": "\u2019",
    "\u00e2\u0080\u009c": "\u201c",
    "\u00e2\u0080\u009d": "\u201d",
    "\u00e2\u0080\u00b2": "\u2032",
    "\u00e2\u0080\u00b3": "\u2033",
    "\u00e2\u0088\u0092": "\u2212",
    "\u00e2\u0089\u0088": "\u2248",
    "\u00e2\u0089\u00a4": "\u2264",
    "\u00e2\u0089\u00a5": "\u2265",
    "\u00e2\u0082\u0080": "\u2080",
    "\u00e2\u0082\u0081": "\u2081",
    "\u00e2\u0082\u0082": "\u2082",
    "\u00e2\u0082\u0083": "\u2083",
    "\u00e2\u0082\u0084": "\u2084",
    "\u00e2\u0082\u0085": "\u2085",
    "\u00e2\u0082\u0086": "\u2086",
    "\u00e2\u0082\u0087": "\u2087",
    "\u00e2\u0082\u0088": "\u2088",
    "\u00e2\u0082\u0089": "\u2089",
    "\u00e2\u20ac\u0090": "\u2010",
    "\u00e2\u20ac\u0091": "\u2011",
    "\u00e2\u20ac\u0092": "\u2012",
    "\u00e2\u20ac\u0093": "\u2013",
    "\u00e2\u20ac\u0094": "\u2014",
    "\u00e2\u20ac\u02dc": "\u2018",
    "\u00e2\u20ac\u2122": "\u2019",
    "\u00e2\u20ac\u0153": "\u201c",
    "\u00e2\u20ac\ufffd": "\u201d",
    "\u00e2\u20ac\u00b2": "\u2032",
    "\u00e2\u20ac\u00b3": "\u2033",
    "\u00c3\u00a2\u00c2\u0080\u00c2\u0090": "\u2010",
    "\u00c3\u00a2\u00c2\u0080\u00c2\u0091": "\u2011",
    "\u00c3\u00a2\u00c2\u0080\u00c2\u0092": "\u2012",
    "\u00c3\u00a2\u00c2\u0080\u00c2\u0093": "\u2013",
    "\u00c3\u00a2\u00c2\u0080\u00c2\u0094": "\u2014",
    "\u00c3\u00a2\u00c2\u0080\u00c2\u0098": "\u2018",
    "\u00c3\u00a2\u00c2\u0080\u00c2\u0099": "\u2019",
    "\u00c3\u00a2\u00c2\u0080\u00c2\u009c": "\u201c",
    "\u00c3\u00a2\u00c2\u0080\u00c2\u009d": "\u201d",
    "\u00c3\u00a2\u00c2\u0080\u00c2\u00b2": "\u2032",
    "\u00c3\u00a2\u00c2\u0080\u00c2\u00b3": "\u2033",
    "\u00c3\u00a2\u00c2\u0082\u00c2\u0080": "\u2080",
    "\u00c3\u00a2\u00c2\u0082\u00c2\u0081": "\u2081",
    "\u00c3\u00a2\u00c2\u0082\u00c2\u0082": "\u2082",
    "\u00c3\u00a2\u00c2\u0082\u00c2\u0083": "\u2083",
    "\u00c3\u00a2\u00c2\u0082\u00c2\u0084": "\u2084",
    "\u00c3\u00a2\u00c2\u0082\u00c2\u0085": "\u2085",
    "\u00c3\u00a2\u00c2\u0082\u00c2\u0086": "\u2086",
    "\u00c3\u00a2\u00c2\u0082\u00c2\u0087": "\u2087",
    "\u00c3\u00a2\u00c2\u0082\u00c2\u0088": "\u2088",
    "\u00c3\u00a2\u00c2\u0082\u00c2\u0089": "\u2089",
}


def mojibake_score(text: str) -> int:
    if not text:
        return 0
    markers = [
        "\u00c2",
        "\u00c3",
        "\u00e2",
        "\ufffd",
    ]
    return sum(text.count(marker) for marker in markers)


def fix_local_mojibake_sequences(text: str) -> str:
    fixed = text
    for bad, good in LOCAL_MOJIBAKE_REPLACEMENTS.items():
        fixed = fixed.replace(bad, good)
    return fixed


def decode_once(text: str, source_encoding: str) -> str | None:
    try:
        return text.encode(source_encoding).decode("utf-8")
    except UnicodeError:
        return None


def fix_text(value: Any) -> Any:
    if not isinstance(value, str) or not MOJIBAKE_RE.search(value):
        return value

    best = fix_local_mojibake_sequences(value)
    best_score = mojibake_score(best)

    candidates = [value, best]
    for source_encoding in ("latin1", "cp1252"):
        decoded = decode_once(value, source_encoding)
        if decoded is not None:
            candidates.append(fix_local_mojibake_sequences(decoded))
            # Some cells have been mojibaked more than once.
            decoded_twice = decode_once(decoded, source_encoding)
            if decoded_twice is not None:
                candidates.append(fix_local_mojibake_sequences(decoded_twice))

    for candidate in candidates:
        score = mojibake_score(candidate)
        if score < best_score:
            best = candidate
            best_score = score

    return best


def fix_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    fixed = df.copy()
    changed = 0
    for col in fixed.columns:
        if fixed[col].dtype != object and not pd.api.types.is_string_dtype(fixed[col].dtype):
            continue
        new_values = []
        for value in fixed[col].tolist():
            new_value = fix_text(value)
            if new_value != value:
                changed += 1
            new_values.append(new_value)
        fixed[col] = new_values
    return fixed, changed


def fix_excel(path: Path, apply: bool) -> dict:
    changed = 0
    rows = 0
    workbook = load_workbook(path)
    for worksheet in workbook.worksheets:
        rows += max(worksheet.max_row - 1, 0)
        for row in worksheet.iter_rows():
            for cell in row:
                if not isinstance(cell.value, str):
                    continue
                fixed_value = fix_text(cell.value)
                if fixed_value != cell.value:
                    changed += 1
                    if apply:
                        cell.value = fixed_value
    if apply and changed:
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False, dir=path.parent) as tmp:
            tmp_path = Path(tmp.name)
        try:
            workbook.save(tmp_path)
            tmp_path.replace(path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    return {"file": str(path), "type": "xlsx", "rows": rows, "changed_cells": changed}


def fix_csv(path: Path, apply: bool) -> dict:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    fixed_df, changed = fix_dataframe(df)
    if apply and changed:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8-sig", newline="", suffix=".csv", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            fixed_df.to_csv(tmp, index=False)
        tmp_path.replace(path)
    return {"file": str(path), "type": "csv", "rows": int(len(df)), "changed_cells": changed}


def fix_json(path: Path, apply: bool) -> dict:
    text = path.read_text(encoding="utf-8")
    fixed_text = fix_text(text)
    changed = int(fixed_text != text)
    if apply and changed:
        path.write_text(fixed_text, encoding="utf-8")
    return {"file": str(path), "type": "json", "changed_text": changed}


def tracked_dataset_files(repo: Path, rel_dir: str) -> list[Path]:
    output = subprocess.check_output(["git", "ls-files", rel_dir], cwd=repo, text=True)
    return [repo / line.strip() for line in output.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix obvious UTF-8 mojibake in tracked EVA dataset files.")
    parser.add_argument("--repo", default=".", help="Repository root.")
    parser.add_argument("--rel-dir", default="data/eva_flame_retardant", help="Tracked dataset directory.")
    parser.add_argument("--apply", action="store_true", help="Write fixed files. Omit for dry-run.")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    results = []
    for path in tracked_dataset_files(repo, args.rel_dir):
        suffix = path.suffix.lower()
        try:
            if suffix == ".xlsx":
                results.append(fix_excel(path, args.apply))
            elif suffix == ".csv":
                results.append(fix_csv(path, args.apply))
            elif suffix == ".json":
                results.append(fix_json(path, args.apply))
            else:
                results.append({"file": str(path), "type": suffix.lstrip("."), "skipped": True})
        except Exception as exc:
            results.append({"file": str(path), "type": suffix.lstrip("."), "error": repr(exc)})

    print(json.dumps({"applied": args.apply, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
