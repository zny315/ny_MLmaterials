# GT50 Extraction Benchmark

This directory is a self-contained replay package for the paper's extraction ablation.

- ground_truth/: publication-clean GT50 workbook.
- extraction_outputs/: four frozen extraction-method outputs.
- evaluator/: hierarchical, order-invariant evaluator and rule tests.
- expected_metrics.json: exact expected extraction metrics.
- release_manifest.json: frozen hashes and benchmark metadata.

The name extraction_outputs is deliberate: these workbooks are structured extraction results, not material-property predictions.

Run from the repository root:

    python scripts/replay_gt50.py --output-dir outputs/gt50_replay

The output directory must not already exist. The replay is offline.