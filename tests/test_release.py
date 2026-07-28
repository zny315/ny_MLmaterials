from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "eva_flame_retardant"


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_release_audit_passes() -> None:
    result = run("scripts/release_audit.py")
    assert result.returncode == 0, result.stdout + result.stderr


def test_manifest_hashes_match() -> None:
    manifest = json.loads(
        (DATA_DIR / "release_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    for filename, metadata in manifest["files"].items():
        path = DATA_DIR / filename
        assert path.is_file()
        assert file_sha256(path) == metadata["sha256"]


def test_canonical_workbook_shapes() -> None:
    manifest = json.loads(
        (DATA_DIR / "release_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    expected = {
        filename: (metadata["rows"], len(metadata["columns"]))
        for filename, metadata in manifest["files"].items()
    }

    for filename, (rows, columns) in expected.items():
        workbook = load_workbook(DATA_DIR / filename, read_only=True)
        try:
            sheet = workbook.active
            headers = [
                "" if value is None else str(value)
                for value in next(sheet.iter_rows(min_row=1, max_row=1, values_only=True))
            ]
            while headers and not headers[-1].strip():
                headers.pop()
            observed_rows = sum(
                any(value is not None and str(value).strip() for value in row)
                for row in sheet.iter_rows(min_row=2, values_only=True)
            )
            assert observed_rows == rows
            assert len(headers) == columns
        finally:
            workbook.close()


def test_supported_cli_help() -> None:
    scripts = [
        "clean_dict.py",
        "eva_aux_performance.py",
        "eva_split_pipeline.py",
        "fix_mojibake.py",
        "literature_ml_baseline.py",
        "LLMvision_extractor.py",
        "loading_conversion_normalizer.py",
        "minerUprocess.py",
        "normalize_exact_duplicates.py",
        "normalize_pipeline.py",
        "schema.py",
    ]
    for name in scripts:
        result = run(str(Path("pipeline") / name), "--help")
        assert result.returncode == 0, f"{name}\n{result.stdout}\n{result.stderr}"


def test_extraction_pipeline_scope() -> None:
    required = {
        "loading_conversion_normalizer.py",
        "normalize_pipeline.py",
        "public_utils.py",
        "schema.py",
    }
    pipeline_dir = ROOT / "pipeline"
    assert all((pipeline_dir / name).is_file() for name in required)
    assert not (pipeline_dir / "performance_reliability_scorer.py").exists()
    assert not (pipeline_dir / "quantify_entity_parts.py").exists()
    assert (ROOT / "benchmark" / "gt50" / "ground_truth").is_dir()
    assert not (ROOT / "benchmark" / "eva10_pipeline_result").exists()
