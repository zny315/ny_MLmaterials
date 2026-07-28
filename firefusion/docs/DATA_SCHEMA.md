# Prepared snapshot contract

The code intentionally starts after document extraction, chemistry parsing,
feature construction, and reliability adjudication. This keeps the released
statistical workflow independent of private raw files and machine-specific
preprocessing paths.

## `features.npz`

Required arrays:

| Key | Shape | Meaning |
|---|---:|---|
| `X_source_full` | `(n_source, p_full)` | Full literature feature matrix used by tree models and pooling |
| `X_source_compact` | `(n_source, p_compact)` | Compact literature matrix used by scaled linear/GP/TabPFN models |
| `X_target_full` | `(n_target, p_full)` | Full in-lab candidate feature matrix |
| `X_target_compact` | `(n_target, p_compact)` | Compact in-lab candidate matrix |
| `source_sample_ids` | `(n_source,)` | Unique literature row identifiers |
| `target_sample_ids` | `(n_target,)` | Unique in-lab formulation identifiers |
| `source_reliability_eligible` | `(n_source,)` | Boolean eligibility mask |

All feature values must be finite. A literature row is used for an endpoint
only when its eligibility flag is true and that endpoint's outcome is finite.
Thus `not_applicable` literature assessments never receive reliability 1.0;
they are excluded from fitting.

## Outcome tables

Provide CSV (preferred for portability) or Parquet files:

- `source_targets.csv` / `.parquet`
- `target_samples.csv` / `.parquet`

Each table contains a unique `sample_id` plus the configured endpoints. For
direct compatibility with the original prepared snapshot, `_key` is also
accepted as the literature-table ID column. The order may differ from the NPZ
order; the loader validates IDs and reorders the tables explicitly.

The supplied configurations use:

- `LOI_rel_increase`: continuous regression target;
- `pHRR_rel_reduction`: continuous regression target;
- `UL94_V0`: binary target encoded as 0/1.

## `split_manifest.csv`

Required columns:

`split_id`, `formulation_split_id`, `target`, `task_type`, `scenario`, `axis`,
`strategy`, `N`, `trial`, `train_sample_ids`, `test_sample_ids`.

The two ID-list columns are JSON arrays in CSV and list columns in Parquet.
Train order and test order are preserved: this matters for TabPFN because the
ordered query batch is estimator-defining. Every selected interpolation split
must have disjoint train/test IDs and cover the same target candidate universe.

Run `python scripts/make_example_data.py --output example_data` for a complete
synthetic example.
