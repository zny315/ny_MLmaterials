# GT50 Ablation Generation

This directory publishes the API-generation code for the four extraction
configurations reported in the paper. The prompt text, three-view schemas, and
post-processing mappings are part of the runner and are protected by SHA-256
snapshot checks.

The V0/V2 prompts and one-shot schemas are readable text snapshots under
`prompts/`. V1 is assembled from the expert pipeline prompts and the frozen
expert three-view schema; V3 uses the same frozen expert pipeline prompts.

| Paper label | Runner configuration | Description |
|---|---|---|
| V0 Direct LLM | `direct_naive_tri_view` | One multimodal request with the schema-only three-view prompt. |
| V1 Expert Prompt Only | `expert_tri_view_single_shot` | One multimodal request with the full expert rules and three-view schema. |
| V2 Task Decomposition Only | `split_naive` | Separate formulation and performance requests with schema-only prompts. |
| V3 Full Pipeline | `full_pipeline` | Separate requests with the frozen expert prompts and canonical merge. |

The default GT50 article list is read from the publication-clean ground-truth
workbook and joined to `data/full_corpus_input_order.csv`. Supply the matching
50 MinerU paper directories under `inputs/mineru/<source_id>/`, then run from
the repository root:

```bash
python benchmark/gt50/generation/eva_gt50_ablation_runner.py \
  --configuration all \
  --input inputs/mineru \
  --output-dir outputs/gt50_generation \
  --model YOUR_MODEL \
  --api-keys-file .secrets/api_keys.txt
```

The output directory must not already exist. Use `--workers 1` for the most
conservative replay. The optional V3 supplemental visual-performance stage is
enabled with `--include-visual-stage`; it uses `temperature=0`, while the text
formulation and performance requests use `temperature=0.1`.

Runtime API responses are written below the selected output directory so a run
can be repaired or audited locally. They are ignored by Git and are not part of
the public release. Model API responses are not expected to be byte-identical
across calls. The checked-in workbooks under `../extraction_outputs/` are the
frozen paper outputs, and `scripts/replay_gt50.py` reproduces their reported
metrics without an API call.
