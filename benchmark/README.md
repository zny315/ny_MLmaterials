# Extraction Benchmark

`gt50/` is the frozen, self-contained replay package for the paper's extraction
ablation. It contains the publication-clean 50-article ground truth, four
structured extraction outputs, the hierarchical evaluator, expected metrics,
and integrity manifests.

Run from the repository root:

```bash
python scripts/replay_gt50.py --output-dir outputs/gt50_replay
```

The replay is offline and does not require source PDFs, MinerU derivatives,
model credentials, or raw API responses. The output directory must not already
exist.
