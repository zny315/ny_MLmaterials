# Code map

| File | Responsibility |
|---|---|
| `config.py` | Configuration loading and invariant checks |
| `data.py` | Prepared-snapshot validation and endpoint filtering |
| `splits.py` | Frozen ordered split-manifest loading |
| `estimators.py` | All lab/literature benchmark backbones and weighted pooling fit |
| `benchmark.py` | Lab-only and literature-only evaluation |
| `affine.py` | Commensurability Bayes factor and constrained positive-affine posterior |
| `synthesis.py` | Deterministic Bayesian convex synthesis |
| `transfer.py` | Strict nested-LOO expert construction and final evaluation |
| `metrics.py` | RMSE, R², AUC, mean and sample SD |
| `reporting.py` | Cross-run/shard metric collation |
| `example.py` | Fully synthetic schema-valid demonstration data |
| `cli.py` | Public command-line interface and deterministic sharding |

The statistical layers are separated from data access and command-line code so
the formulas can be tested without loading project artifacts or a GPU model.
