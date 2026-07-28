# FireFusion benchmark and Bayesian transfer

This directory is a self-contained, publication-oriented implementation of the
workflow used to benchmark lab-only and literature-only predictors and to fit
the final Bayesian two-expert transfer model for flame-retardant formulation
data.

The public boundary starts from a **prepared numeric snapshot**. Raw papers,
LLM extraction files, proprietary lab spreadsheets, and machine-specific paths
are deliberately outside this package. See `docs/DATA_SCHEMA.md` for the exact
input contract.

## Data availability

The implementation, formal configuration, and synthetic example generator are
included. The prepared snapshot used for the paper is awaiting upload by the
data maintainer and is therefore not yet present under `data/`.

This distinction is deliberate: `scripts/make_example_data.py` verifies the
software path but must not be used as a substitute for the paper data or its
reported results. Data maintainers should follow
`docs/DATA_RELEASE_CHECKLIST.md` before announcing full reproducibility.

## What is included

1. Frozen split-manifest validation.
2. Lab-only and literature-only benchmarks for regression and classification.
3. A fixed, benchmark-selected backbone for each regression endpoint.
4. Strict nested leave-one-out construction of two transfer experts:
   commensurability-weighted pooling and Bayesian positive-affine calibration.
5. Bayesian convex synthesis of the two experts.
6. Trial-level predictions, metrics, diagnostics, and publication-ready summary
   CSV tables.

UL-94 is evaluated as a classification benchmark (ROC AUC). Transfer is applied
only to LOI and pHRR regression, matching the reported study.

## Install

```bash
conda env create -f environment.yml
conda activate firefusion
python -m pip install -e .
```

TabPFN is optional because it may require a CUDA-enabled PyTorch installation:

```bash
python -m pip install -e '.[tabpfn]'
```

## End-to-end smoke test

The smoke configuration uses only scikit-learn models, so it runs on CPU:

```bash
python scripts/make_example_data.py --output example_data
firefusion all --config configs/smoke.json --data example_data --output smoke_output
pytest -q
```

## Reproduce the paper workflow

After the data maintainer places the validated prepared snapshot under `data/`,
run:

```bash
firefusion benchmark --config configs/paper.json --data data --output results/benchmark
firefusion transfer  --config configs/paper.json --data data --output results/transfer
firefusion summarize --inputs results/benchmark results/transfer --output results/summary
```

The paper configuration evaluates the frozen trials 8--39 and fixes one lab
and one literature backbone per endpoint across every shot count; it never
chooses a different comparator after seeing a particular `N` result.

### Eight-GPU execution

The formal jobs use deterministic trial-modulo sharding. Launch one process per
GPU, changing both indices from 0 through 7:

```bash
CUDA_VISIBLE_DEVICES=0 firefusion all --config configs/paper.json --data data \
  --output results/shard-000 --tabpfn-device cuda:0 --shard-index 0 --shard-count 8
```

Repeat for `CUDA_VISIBLE_DEVICES=1,...,7` and `--shard-index 1,...,7`. Each
process writes an independent directory, so no concurrent writer shares an
artifact. `firefusion summarize` accepts all shard-level benchmark and transfer
directories.

## Main outputs

- `predictions.csv`: one row per split, sample, and method.
- `split_metrics.csv`: RMSE/R2 for regression or ROC AUC for classification.
- `summary_metrics.csv`: mean, sample SD, and trial count by scenario and method.
- `diagnostics.jsonl`: fitted commensurability powers, affine posterior details,
  and synthesis weights for every transfer split.
- `run_metadata.json`: config copy, versions, timestamps, and input checksums.

The commands refuse to overwrite a nonempty output directory unless
`--overwrite` is supplied.

## Citation and license

See `CITATION.cff`. Code is released under the MIT License.
