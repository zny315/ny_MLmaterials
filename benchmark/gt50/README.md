# GT50 Extraction Benchmark

This directory is a self-contained replay package for the paper's extraction ablation.

- ground_truth/: publication-clean GT50 workbook.
- extraction_outputs/: four frozen extraction-method outputs.
- generation/: exact V0-V3 prompts, schemas, post-processing, and API runner.
- evaluator/: hierarchical, order-invariant evaluator and rule tests.
- expected_metrics.json: exact expected extraction metrics.
- release_manifest.json: frozen hashes and benchmark metadata.

The name extraction_outputs is deliberate: these workbooks are structured extraction results, not material-property predictions.

Run from the repository root:

    python scripts/replay_gt50.py --output-dir outputs/gt50_replay

The output directory must not already exist. The replay is offline.

To generate new API outputs with the four paper configurations, see
[generation/README.md](generation/README.md). API responses can vary across
calls; the frozen workbooks remain the reference artifacts for exact metric
replay.
