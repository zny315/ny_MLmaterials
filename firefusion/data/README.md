# Prepared paper snapshot

This directory is the publication-data landing zone for FireFusion. The formal
paper snapshot is awaiting upload by the data maintainer. No synthetic file in
this repository should be renamed or presented as the paper snapshot.

## Files expected from the data maintainer

```text
data/
├── features.npz
├── source_targets.csv          # or source_targets.parquet
├── target_samples.csv          # or target_samples.parquet
├── split_manifest.csv          # or split_manifest.parquet
├── snapshot_manifest.json
└── README.md
```

The array, table, endpoint, and split fields are defined in
`../docs/DATA_SCHEMA.md`. The upload, licensing, count-reconciliation, and
checksum requirements are defined in
`../docs/DATA_RELEASE_CHECKLIST.md`.

## Expected formal endpoints

- `LOI_rel_increase`: regression;
- `pHRR_rel_reduction`: regression;
- `UL94_V0`: binary classification.

The uploader must record the final number of target formulations in
`snapshot_manifest.json` and reconcile it with the manuscript before release.
In particular, the current 37-versus-38 wording inconsistency must not be
resolved by silently dropping or duplicating a row.

## After upload

From the repository root:

```bash
firefusion benchmark --config firefusion/configs/paper.json --data firefusion/data --output results/benchmark
firefusion transfer --config firefusion/configs/paper.json --data firefusion/data --output results/transfer
firefusion summarize --inputs results/benchmark results/transfer --output results/summary
```

Raw literature files, raw laboratory spreadsheets, direct identifiers,
credentials, and machine-specific paths must remain outside this directory.
