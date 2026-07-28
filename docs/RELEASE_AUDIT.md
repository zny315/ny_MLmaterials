# Release Scope Audit

Basis: the current KDD 2027 manuscript "Extract Broadly, Calibrate Locally: Formulation-Aware Multi-Fidelity Learning for Flame-Retardant Polymers."

This repository is intentionally the extraction-and-normalization companion, not a release of every manuscript component.

| Paper artifact | Status in this repository | Verification |
|---|---|---|
| MinerU preprocessing | Included | CLI and compile check |
| Formulation and entity-part extraction | Included | Latest parser and thread-safety test |
| Roster-conditioned text performance extraction | Included | Latest parser and compile check |
| Visual supplementation and merge | Included | Latest implementation and compile check |
| Deterministic normalization | Included | Unified normalization entry point and tests |
| Optional API-assisted SMILES enrichment | Included | Explicit flag; runtime outputs ignored |
| Three canonical 535-paper long tables | Included | 17,150/5,687/5,786 rows |
| Derived manual-review wide table | Included | 5,417 rows, 147 columns; rebuilt from the three long tables; not a modeling input |
| Historical 545-input order | Included | Path-free manifest with indices 1-545 |
| GT50 clean ground truth | Included | 2,828/1,134/721 rows; 50 articles |
| Four GT50 extraction outputs | Included | Per-method manifests and hashes |
| GT50 evaluator | Included | Exact offline replay |
| Raw API response and run-result JSON | Excluded from repository | Runtime generation retained; path scan enforced |
| Controlled T01-T38 laboratory data | Excluded by scope | Separate downstream release |
| Property-prediction and calibration code | Excluded by scope | Separate downstream release |
| Trial-level downstream predictions | Excluded by scope | Separate downstream release |

The Full Pipeline frozen extraction output replays F1 values of 0.9175698916, 0.7574338086, 0.9617958728, and 0.8876771044 for entities, parts, performance, and overall.

T01-T38 are 38 controlled EVA formulations and are not a one-to-one list of 38 mineral species.
