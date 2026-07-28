# Prepared snapshot release checklist

This checklist is for the maintainer who uploads the formal FireFusion paper
snapshot. It defines the handoff boundary without requiring raw papers,
private extraction artifacts, or proprietary laboratory workbooks.

## 1. Materialize the public files

Provide:

- `features.npz`;
- `source_targets.csv` or `source_targets.parquet`;
- `target_samples.csv` or `target_samples.parquet`;
- `split_manifest.csv` or `split_manifest.parquet`;
- `snapshot_manifest.json`.

Do not upload raw papers, page images, raw laboratory spreadsheets, model
credentials, machine-specific paths, or direct personal identifiers.

## 2. Reconcile the scientific record

Before upload:

- resolve the manuscript's 37-versus-38 target-formulation inconsistency;
- confirm every `target_sample_id` has exactly one target row;
- record any excluded formulation and the exclusion rule;
- confirm endpoint units and transformations;
- confirm that the frozen split manifest covers the declared candidate
  universe without train/test overlap;
- confirm that the formal config's trial range and shot counts match the paper.

Never alter the sample count merely to match prose. Correct the manuscript or
the prepared snapshot using the underlying experimental record.

## 3. Write `snapshot_manifest.json`

The manifest should contain at least:

```json
{
  "snapshot_version": "1.0.0",
  "created_utc": "YYYY-MM-DDTHH:MM:SSZ",
  "source_rows": 0,
  "target_rows": 0,
  "full_feature_count": 0,
  "compact_feature_count": 0,
  "targets": {
    "LOI_rel_increase": "reg",
    "pHRR_rel_reduction": "reg",
    "UL94_V0": "clf"
  },
  "license": "<SPDX identifier or access terms>",
  "files": {
    "features.npz": {"sha256": "<hex>", "bytes": 0}
  }
}
```

Add every released data file to `files` with its SHA-256 digest and byte size.
Also record the feature-construction code revision, data-freeze date, exclusion
rules, de-identification statement, and contact or access procedure.

## 4. Validate locally

Install the repository development environment and run:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
python scripts/release_audit.py
```

Then run the formal benchmark and transfer commands from `../README.md`. Check
that output metadata record the config hash, input checksums, software
versions, split count, and shard settings.

## 5. Review before publication

- inspect `git diff --stat` and confirm no raw or private files are staged;
- run a secret scan and inspect unusually large files;
- verify the declared license and contributor authorization;
- compare summary tables against the paper;
- update the root data card, reproducibility guide, release-readiness document,
  and citation metadata;
- tag an immutable version and archive it with a persistent identifier.
