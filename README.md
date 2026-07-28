# Extract Broadly, Calibrate Locally

[![CI](https://github.com/zny315/ny_MLmaterials/actions/workflows/ci.yml/badge.svg)](https://github.com/zny315/ny_MLmaterials/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg)](DATA_LICENSE.md)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](requirements.txt)

Formulation-aware multimodal extraction of flame-retardant polymer data from
scientific literature. This repository contains the extraction pipeline,
curated EVA literature tables, the frozen GT50 extraction benchmark, a
leakage-aware literature baseline, and the FireFusion
Bayesian benchmark-and-transfer implementation used by the companion
manuscript:

> **Extract Broadly, Calibrate Locally: Formulation-Aware Multi-Fidelity
> Learning for Flame-Retardant Polymers**

The release favors auditable, linked long tables over a single denormalized
spreadsheet. It preserves formulation entities, entity-part relationships,
performance records, and provenance.

## Release scope

| Component | Status |
|---|---|
| Formulation, performance, and visual extraction pipeline | Included |
| Curated literature-derived EVA tables from 535 articles | Included |
| Deterministic normalization and documented loading conversion | Included |
| Frozen GT50 extraction benchmark and evaluator | Included |
| Literature-only, article-grouped ML baseline | Included |
| FireFusion Bayesian benchmark and transfer implementation | Included |
| Publisher PDFs and MinerU full-text derivatives | Not redistributed |
| Prepared target-laboratory snapshot and frozen split manifest | Pending data-maintainer upload |

FireFusion's synthetic CPU smoke test is fully runnable now. Formal paper
benchmarks require the prepared numeric snapshot documented under
`firefusion/data/`; the repository must not be described as reproducing the
reported paper results until that snapshot is uploaded and its sample count is
reconciled with the manuscript. See
[Reproducibility](docs/REPRODUCIBILITY.md).

## Repository layout

```text
.
|-- pipeline/       Extraction, normalization, auxiliary, and baseline scripts
|-- data/           Curated structured release tables and integrity manifest
|-- benchmark/      Frozen GT50 extraction benchmark and evaluator
|-- firefusion/     Bayesian benchmark and multi-fidelity transfer package
|-- test_data/      Regression workbooks used by existing repository components
|-- tests/          Offline tests; no API credentials or network required
|-- scripts/        Release-integrity audit
`-- docs/           Release-scope and reproducibility notes
```

## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/zny315/ny_MLmaterials.git
cd ny_MLmaterials
python -m venv .venv
```

Activate the environment:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e ./firefusion
```

MinerU is an external optional dependency needed only to preprocess PDFs.
Install it according to its upstream documentation and ensure `mineru` or
`magic-pdf` is available on `PATH`.

## Quick validation

All release checks are offline and do not call a model API:

```bash
python -m compileall -q pipeline scripts tests firefusion/src firefusion/scripts firefusion/tests
python scripts/release_audit.py
python scripts/replay_gt50.py --output-dir outputs/gt50_replay
python -m pytest
```

## Data

The public literature snapshot is in `data/eva_flame_retardant/`.

| File | Rows | Purpose |
|---|---:|---|
| `formulation_entities.xlsx` | 17,150 | One material entity in one sample |
| `formulation_entity_parts.xlsx` | 5,687 | Internal parts and relations for complex entities |
| `performance_samples_final.xlsx` | 5,786 | Sample-linked performance records |
| `flame_retardant_data_optimized_final.xlsx` | 5,417 | Convenience wide table for manual review |
| `release_manifest.json` | 1 | Counts, column order, and SHA-256 checksums |

The extraction code uses canonical English field names. `pipeline/schema.py`
maps legacy workbook headers at load time. Use `sample_uid` to join formulation
and performance tables, `performance_record_uid` as the stable record identity,
and `article_index` for article-grouped splitting.

The structured release is documented in [DATA_CARD.md](DATA_CARD.md). The MIT
license covers software; the repository's authored structured data are
released under [CC BY 4.0](DATA_LICENSE.md). Third-party article text and
figures are not covered and are not redistributed on this branch.

## Pipeline

The public extraction workflow is:

```text
lawfully obtained PDFs
    -> MinerU preprocessing
    -> formulation extraction
    -> text/table performance extraction
    -> visual performance supplementation
    -> linked canonical long tables
    -> deterministic normalization and SMILES enrichment
    -> manual-review wide table
```

### 1. Preprocess documents

Place lawfully obtained PDFs under `inputs/pdfs/` (ignored by Git):

```bash
python pipeline/minerUprocess.py \
  --pdf-dir inputs/pdfs \
  --output-dir inputs/mineru
```

### 2. Configure a compatible multimodal endpoint

Credentials and endpoint settings are read from environment variables. Never
commit credentials.

```bash
export OPENAI_BASE_URL="https://api.openai.com/v1/chat/completions"
export OPENAI_MODEL="<multimodal-model-id>"
export OPENAI_API_KEY="<secret>"
```

PowerShell:

```powershell
$env:OPENAI_BASE_URL = "https://api.openai.com/v1/chat/completions"
$env:OPENAI_MODEL = "<multimodal-model-id>"
$env:OPENAI_API_KEY = "<secret>"
```

### 3. Run extraction

Generate a request preview without a network call:

```bash
python pipeline/eva_split_pipeline.py \
  --task all \
  --input inputs/mineru \
  --output-dir outputs/eva \
  --limit 1 \
  --dry-run
```

Run the staged workflow:

```bash
python pipeline/eva_split_pipeline.py \
  --task all \
  --input inputs/mineru \
  --output-dir outputs/eva \
  --model gpt-5.5 \
  --workers 3 \
  --api-retries 4 \
  --retry-base-delay 20 \
  --read-timeout 540 \
  --max-tokens 65536 \
  --stream
```

The tasks `formulation`, `performance`, `performance_v`, `merge`, and `wide`
can also run independently. Fresh API responses are not expected to be
bit-for-bit identical across providers or model revisions.

### 4. Normalize canonical tables

```bash
python pipeline/normalize_pipeline.py \
  --input-dir outputs/eva \
  --output-dir outputs/eva_normalized
```

This stage repairs text encoding, removes exact duplicate entity rows, applies
the expert material dictionary, and fills missing canonical wt% or phr fields
only when a complete documented formulation basis permits conversion. Existing
normalized loading values are preserved. Add `--visual-smiles` to opt into
image-assisted SMILES completion.

### 5. Rebuild the manual-review wide table

```bash
python pipeline/eva_split_pipeline.py \
  --task wide \
  --output-dir outputs/eva_normalized \
  --merged-wide outputs/eva_normalized/flame_retardant_data_optimized_final.xlsx
```

The wide table is a deterministic convenience view for human auditing and is
not a modeling input.

### 6. Runtime JSON policy

The extraction code writes raw response JSON, caches, summaries, and replay
metadata under ignored runtime output directories. These files support local
resume and audit workflows but are not committed to the public repository.

## Bayesian benchmark and transfer

`firefusion/` is a self-contained implementation of the lab-only and
literature-only benchmarks, commensurability-weighted pooling, constrained
Bayesian positive-affine calibration, and Bayesian two-expert synthesis.

Run the complete CPU-only synthetic example from the repository root:

```bash
python firefusion/scripts/make_example_data.py --output outputs/firefusion-example
firefusion all \
  --config firefusion/configs/smoke.json \
  --data outputs/firefusion-example \
  --output outputs/firefusion-smoke
```

For the formal configuration, the data maintainer must first populate
`firefusion/data/` according to the
[prepared snapshot contract](firefusion/docs/DATA_SCHEMA.md) and
[release checklist](firefusion/docs/DATA_RELEASE_CHECKLIST.md). Then run:

```bash
firefusion benchmark \
  --config firefusion/configs/paper.json \
  --data firefusion/data \
  --output results/benchmark

firefusion transfer \
  --config firefusion/configs/paper.json \
  --data firefusion/data \
  --output results/transfer
```

The code and formal configuration are versioned now; the prepared paper
snapshot is intentionally not represented by synthetic placeholders.

## Benchmark

`benchmark/gt50/` contains the publication-clean ground truth, four frozen
extraction outputs, the hierarchical evaluator, expected metrics, and integrity
manifests for the paper's extraction ablation. Replay it offline with:

```bash
python scripts/replay_gt50.py --output-dir outputs/gt50_replay
```

The replay regenerates the metric tables from the checked-in structured
workbooks; it does not call a model API. See [benchmark/README.md](benchmark/README.md).

## Citation

Citation metadata are available in [CITATION.cff](CITATION.cff). Until the
paper has a persistent identifier, cite the repository and the exact commit
used. After publication, update both files with the final venue, year, and DOI.

## Contributing and security

Bug reports and focused pull requests are welcome. Read
[CONTRIBUTING.md](CONTRIBUTING.md) before contributing and report suspected
credential exposure privately using [SECURITY.md](SECURITY.md).

## License

- Code and documentation: [MIT](LICENSE)
- Authored structured data and annotations: [CC BY 4.0](DATA_LICENSE.md)
- Third-party publications: governed by their respective rights holders and
  excluded from this release branch
