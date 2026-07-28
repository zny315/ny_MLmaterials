# Reproducibility Guide

## Supported Scope

The public-release snapshot supports:

1. offline validation of the curated literature tables and their hashes;
2. local PDF-to-MinerU preprocessing for lawfully obtained documents;
3. formulation, performance, and visual-supplement extraction;
4. deterministic normalization and documented loading conversion;
5. reconstruction of the manual-review wide table;
6. exact offline replay of the frozen GT50 extraction ablation;
7. a CPU synthetic smoke test of the FireFusion workflow.

The extraction release is complete without raw model responses: hosted calls
are nondeterministic, while the checked-in canonical tables and frozen GT50
outputs are the auditable reference artifacts.

The formal FireFusion configuration is included under `firefusion/`. Its
prepared target-laboratory snapshot and formal split manifest remain a separate
data-release dependency.

## Environment

- Python: 3.10-3.12
- Dependencies: `requirements-dev.txt`
- Optional document parser: MinerU CLI
- Optional model service: an OpenAI-compatible chat-completions endpoint with
  multimodal input support

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

## Verify the Checked-In Release

```bash
python -m compileall -q pipeline scripts tests firefusion/src firefusion/scripts firefusion/tests
python scripts/release_audit.py
python scripts/replay_gt50.py --output-dir outputs/gt50_replay
python -m pytest
```

The release audit checks manifest hashes, workbook shapes and English headers,
common secret formats, credential filenames, machine-specific paths, non-English
text and emoji in the public extraction assets, tracked runtime-response JSON,
and excluded source-document derivatives. These checks are offline and do not
call a model API.

The GT50 replay regenerates metrics from the frozen ground truth and four
structured extraction outputs. The expected Full Pipeline F1 values are
0.9175698916 for entities, 0.7574338086 for parts, 0.9617958728 for performance,
and 0.8876771044 overall.

## Regenerate Literature Extraction

1. Obtain source papers through lawful channels.
2. Store PDFs under `inputs/pdfs/`.
3. Run `pipeline/minerUprocess.py` to create local MinerU derivatives.
4. Set `OPENAI_BASE_URL`, `OPENAI_MODEL`, and `OPENAI_API_KEY`.
5. Run `pipeline/eva_split_pipeline.py --task all`.
6. Run `pipeline/normalize_pipeline.py` on the extracted canonical tables.
7. Rebuild the manual-review wide table with
   `pipeline/eva_split_pipeline.py --task wide`.

Input and output directories are ignored by Git. Preserve the provider and
model revision, request parameters, timestamps, source-document checksums,
environment lock, exact code commit, and raw responses locally when an audit
trail is required.

Fresh API responses need not match the frozen release byte for byte. Compare
schema validity, stable identifiers, evidence alignment, canonical table
semantics, and benchmark metrics.

## Runtime JSON Policy

The extraction pipeline continues to write raw response JSON and replay caches
for local resume and debugging. These runtime artifacts are intentionally
excluded from Git; only release manifests and GT50 metric metadata are tracked
as JSON.

## Reproduce Bayesian Transfer

The formal code path is documented in `firefusion/README.md`. Its public input
boundary is a prepared numeric snapshot rather than raw laboratory spreadsheets.
The data maintainer must provide:

- `features.npz`;
- `source_targets.csv` or `.parquet`;
- `target_samples.csv` or `.parquet`;
- `split_manifest.csv` or `.parquet`;
- `snapshot_manifest.json` with file hashes, row counts, feature dimensions,
  endpoint definitions, creation provenance, and license.

The exact field and array contract is in `firefusion/docs/DATA_SCHEMA.md`; the
publication handoff procedure is in
`firefusion/docs/DATA_RELEASE_CHECKLIST.md`. Once those files are present:

```bash
firefusion benchmark --config firefusion/configs/paper.json --data firefusion/data --output results/benchmark
firefusion transfer --config firefusion/configs/paper.json --data firefusion/data --output results/transfer
firefusion summarize --inputs results/benchmark results/transfer --output results/summary
```

## Publication Notes

- Update citation metadata with the final venue and persistent DOI.
- Archive the exact public commit in a long-term repository.
- Confirm that third-party article content is absent from every public ref, not
  only from the current branch, before making a repository-wide rights claim.
