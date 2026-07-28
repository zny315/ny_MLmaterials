import argparse
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from public_utils import console_print as print, install_ascii_exception_hook
except ModuleNotFoundError:
    from pipeline.public_utils import console_print as print, install_ascii_exception_hook

install_ascii_exception_hook()


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DEFAULT_PDF_DIR = REPO_ROOT / "inputs" / "pdf"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "inputs" / "mineru"
DEFAULT_BACKEND = "pipeline"
DEFAULT_METHOD = "auto"


def find_cli(preferred: str | None = None) -> str:
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(["mineru", "magic-pdf"])

    for candidate in candidates:
        path = shutil.which(candidate)
        if path:
            return path

    raise RuntimeError(
        "MinerU command was not found. Install MinerU and make sure either "
        "'mineru' or 'magic-pdf' is available in PATH."
    )


def find_pdf_files(pdf_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted(path for path in pdf_dir.glob(pattern) if path.is_file())


def has_mineru_output(paper_output_dir: Path) -> bool:
    auto_dir = paper_output_dir / "auto"
    target_dir = auto_dir if auto_dir.is_dir() else paper_output_dir
    if not target_dir.is_dir():
        return False

    return any(
        path.is_file() and path.suffix.lower() in {".md", ".txt"}
        for path in target_dir.iterdir()
    )


def build_command(
    cli_path: str,
    pdf_path: Path,
    output_dir: Path,
    backend: str,
    method: str,
    lang: str | None,
) -> list[str]:
    cli_name = Path(cli_path).name.lower()

    if cli_name.startswith("mineru"):
        command = [cli_path, "-p", str(pdf_path), "-o", str(output_dir)]
        if method:
            command.extend(["-m", method])
        if backend:
            command.extend(["-b", backend])
        if lang:
            command.extend(["-l", lang])
        return command

    if cli_name.startswith("magic-pdf"):
        command = [cli_path, "-p", str(pdf_path), "-o", str(output_dir)]
        if method:
            command.extend(["-m", method])
        if lang:
            command.extend(["-l", lang])
        return command

    raise RuntimeError(f"Unsupported MinerU command: {cli_path}")


def process_pdf(
    cli_path: str,
    pdf_path: Path,
    output_dir: Path,
    backend: str,
    method: str,
    lang: str | None,
    force: bool,
) -> str:
    paper_output_dir = output_dir / pdf_path.stem
    if has_mineru_output(paper_output_dir) and not force:
        print(f"[skip] {pdf_path.name} -> output already exists")
        return "skipped"

    command = build_command(cli_path, pdf_path, output_dir, backend, method, lang)
    print(f"[run] {pdf_path.name}")
    print("      " + subprocess.list2cmdline(command))

    result = subprocess.run(command, cwd=str(BASE_DIR))
    if result.returncode != 0:
        print(f"[fail] {pdf_path.name} -> MinerU exited with {result.returncode}")
        return "failed"

    if has_mineru_output(paper_output_dir):
        print(f"[ok] {pdf_path.name} -> {paper_output_dir / 'auto'}")
        return "ok"

    print(f"[warn] {pdf_path.name} finished, but expected output was not found: {paper_output_dir}")
    return "missing_output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch process PDF papers with local MinerU into input_paper/<paper>/auto "
            "Markdown plus images format."
        )
    )
    parser.add_argument(
        "--pdf-dir",
        default=str(DEFAULT_PDF_DIR),
        help=f"Folder containing PDF papers. Default: {DEFAULT_PDF_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"MinerU output folder. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--cli",
        default=None,
        help="Optional MinerU command path/name. Defaults to auto-detect mineru or magic-pdf.",
    )
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        help="Backend passed to the new mineru CLI with -b. Default: pipeline.",
    )
    parser.add_argument(
        "--method",
        default=DEFAULT_METHOD,
        help="Method passed to the old magic-pdf CLI with -m. Default: auto.",
    )
    parser.add_argument(
        "--lang",
        default=None,
        help="Optional document language passed with -l, for example ch or en.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search PDF files recursively under --pdf-dir.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess papers even if input_paper/<paper>/auto already has a markdown file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N PDFs after sorting. Default: 0 means no limit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pdf_dir = Path(args.pdf_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not pdf_dir.is_dir():
        print(f"PDF folder does not exist: {pdf_dir}")
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        cli_path = find_cli(args.cli)
    except RuntimeError as exc:
        print(str(exc))
        return 2

    pdf_files = find_pdf_files(pdf_dir, args.recursive)
    if args.limit and args.limit > 0:
        pdf_files = pdf_files[: args.limit]

    if not pdf_files:
        print(f"No PDF files found in: {pdf_dir}")
        return 0

    print("=" * 60)
    print("MinerU Batch PDF Processor")
    print("=" * 60)
    print(f"PDF folder : {pdf_dir}")
    print(f"Output dir : {output_dir}")
    print(f"MinerU CLI : {cli_path}")
    print(f"PDF count  : {len(pdf_files)}")
    print("=" * 60)

    counts = {"ok": 0, "skipped": 0, "failed": 0, "missing_output": 0}
    for index, pdf_path in enumerate(pdf_files, start=1):
        print(f"\n[{index}/{len(pdf_files)}] {pdf_path}")
        status = process_pdf(
            cli_path=cli_path,
            pdf_path=pdf_path,
            output_dir=output_dir,
            backend=args.backend,
            method=args.method,
            lang=args.lang,
            force=args.force,
        )
        counts[status] += 1

    print("\n" + "=" * 60)
    print("Done")
    print("=" * 60)
    print(f"OK            : {counts['ok']}")
    print(f"Skipped       : {counts['skipped']}")
    print(f"Failed        : {counts['failed']}")
    print(f"Missing output: {counts['missing_output']}")

    if counts["failed"] or counts["missing_output"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
