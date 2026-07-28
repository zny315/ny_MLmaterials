from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from clean_dict import clean_dataframe, read_table, write_table
from fix_mojibake import fix_excel
from loading_conversion_normalizer import normalize_entity_loading_dataframe
from normalize_exact_duplicates import normalize_file
from public_utils import console_print as print
from public_utils import install_ascii_exception_hook


REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = REPO_ROOT / "outputs" / "pipeline"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "pipeline_normalized"
CANONICAL_TABLES = (
    "formulation_entities.xlsx",
    "formulation_entity_parts.xlsx",
    "performance_samples_final.xlsx",
)
ENTITY_TABLES = (
    "formulation_entities.xlsx",
    "formulation_entity_parts.xlsx",
)
def copy_canonical_tables(input_dir: Path, output_dir: Path, force: bool) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in CANONICAL_TABLES:
        source = input_dir / name
        destination = output_dir / name
        if not source.exists():
            if name == "performance_samples_final.xlsx":
                print(f"[SKIP] Optional table not found: {source}")
                continue
            raise FileNotFoundError(f"Required extraction table not found: {source}")
        if destination.exists() and not force:
            raise FileExistsError(f"Output exists: {destination}. Use --force to replace canonical tables.")
        shutil.copy2(source, destination)
        copied.append(destination)
        print(f"[OK] Copied {source.name}")
    return copied


def run_deterministic_stages(output_dir: Path, copied: list[Path], skip_dedupe: bool, skip_dictionary: bool) -> None:
    for path in copied:
        result = fix_excel(path, apply=True)
        print(f"[OK] Mojibake repair: {path.name}; changed_cells={result['changed_cells']}")

    if not skip_dedupe:
        for name in ENTITY_TABLES:
            path = output_dir / name
            result, _ = normalize_file(path, apply=True)
            print(f"[OK] Exact duplicate normalization: {name}; removed={result['duplicate_rows']}")

    if not skip_dictionary:
        for name in ENTITY_TABLES:
            path = output_dir / name
            dataframe = clean_dataframe(read_table(path))
            write_table(dataframe, path)
            print(f"[OK] Dictionary normalization: {name}; rows={len(dataframe)}")

    entities_path = output_dir / "formulation_entities.xlsx"
    entities, changed = normalize_entity_loading_dataframe(read_table(entities_path))
    write_table(entities, entities_path)
    print(
        "[OK] Fill-only loading conversion normalization: "
        + ", ".join(f"{column}={count}" for column, count in changed.items())
    )


def run_command(arguments: list[str]) -> None:
    print("[RUN] " + " ".join(arguments))
    completed = subprocess.run(arguments, check=False)
    if completed.returncode:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(arguments)}")


def run_visual_stage(args: argparse.Namespace, output_dir: Path) -> None:
    if not args.image_folder:
        raise ValueError("--visual-smiles requires --image-folder")
    for name in ENTITY_TABLES:
        source = output_dir / name
        temporary = output_dir / f".{source.stem}_visual.xlsx"
        command = [
            sys.executable,
            str(PIPELINE_DIR / "LLMvision_extractor.py"),
            "--input",
            str(source),
            "--image-folder",
            str(args.image_folder),
            "--output",
            str(temporary),
        ]
        if args.api_base:
            command.extend(["--api-base", args.api_base])
        if args.api_key_file:
            command.extend(["--api-key-file", str(args.api_key_file)])
        if args.model:
            command.extend(["--model", args.model])
        run_command(command)
        temporary.replace(source)
        print(f"[OK] Visual SMILES enrichment: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run extraction-table normalization with optional API-assisted stages.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="Replace canonical tables already present in the output directory.")
    parser.add_argument("--skip-dedupe", action="store_true")
    parser.add_argument("--skip-dictionary", action="store_true")
    parser.add_argument("--visual-smiles", action="store_true", help="Run optional API-assisted visual SMILES enrichment.")
    parser.add_argument("--image-folder", type=Path, default=None)
    parser.add_argument("--api-key-file", type=Path, default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--model", default=None)
    return parser.parse_args()


def main() -> int:
    install_ascii_exception_hook()
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if input_dir == output_dir:
        raise ValueError("Input and output directories must differ.")

    copied = copy_canonical_tables(input_dir, output_dir, args.force)
    run_deterministic_stages(output_dir, copied, args.skip_dedupe, args.skip_dictionary)
    if args.visual_smiles:
        run_visual_stage(args, output_dir)
    print(f"[OK] Normalization complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
