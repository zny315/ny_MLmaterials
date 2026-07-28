# Data Directory

The public release contains:

- `eva_flame_retardant/`: three authoritative path-sanitized long tables, one 5,417-row derived wide table for manual inspection, and `release_manifest.json`.
- `full_corpus_input_order.csv`: the path-free order of the 545 historical extraction inputs.

The wide table is deterministically reconstructed from the three long tables and is not a modeling input. The controlled T01-T38 laboratory data, visual intermediate table, image task index, raw API responses, caches, and run-result JSON are intentionally excluded.
