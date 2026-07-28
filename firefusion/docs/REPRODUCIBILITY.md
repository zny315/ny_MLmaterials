# Reproducibility notes

The code and configurations described here are released. The formal prepared
snapshot is awaiting upload under `../data/`; until then, only the synthetic
CPU smoke workflow can be reproduced from the repository alone.

- Splits are data, not random seeds: the code consumes explicit ordered sample
  IDs from a manifest and never regenerates evaluation splits.
- The selected lab/literature backbone is fixed per endpoint across all `N`.
- Random forests, HistGB, K-fold OOF prediction, Sobol draws, and TabPFN expose
  fixed seeds in the code or configuration.
- For transductive TabPFN inference, the exact ordered outer-test batch is used;
  it is never replaced by a convenience query over all target candidates.
- Pooling is membership-canonical because HistGB is row-order invariant.
- The fixed response normalization uses the selected literature scores over
  the unlabeled target candidate pool. It consumes no target outcomes.
- Every output directory contains a config hash, input checksums, software
  versions, predictions, split-level metrics, and (for transfer) fitted
  diagnostics.

The paper configuration assumes access to TabPFN 2.0.9 and a compatible CUDA
environment. The smoke configuration replaces TabPFN backbones with Ridge/RF
so the entire code path is testable on a small CPU machine.
