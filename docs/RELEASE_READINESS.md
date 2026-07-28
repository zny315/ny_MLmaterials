# Public-Release Readiness

## Completed in This Branch

- the current branch preserves the existing downstream project structure while
  adding the paper's extraction and normalization release;
- source PDFs, MinerU full-text derivatives, raw API responses, caches, and
  run-result JSON are excluded from the tracked extraction assets;
- the three canonical 535-paper long tables use English headers and contain
  17,150 formulation-entity rows, 5,687 entity-part rows, and 5,786 performance
  rows;
- the 5,417-row, 147-column wide table is rebuilt from the three long tables and
  is documented as a manual-review artifact rather than a modeling input;
- the frozen GT50 ground truth, four extraction outputs, evaluator, expected
  metrics, and file hashes are included for offline replay;
- normalization is exposed through one entry point for encoding repair, exact
  duplicate removal, expert dictionary normalization, SMILES enrichment, and
  fill-only loading conversion;
- environment-variable configuration, dependency metadata, tests, CI, release
  audit, citation, licenses, data card, contribution guide, and security policy
  are present;
- the existing FireFusion implementation and its synthetic tests remain in the
  repository without being modified by the extraction merge.

## Remaining Repository-Level Actions

1. Update `CITATION.cff` after acceptance or publication.
2. Create an immutable archival release and record its DOI.
3. Complete the separate data handoff required by the FireFusion formal
   configuration.

## Repository-History Warning

Removing third-party files from this branch does not remove them from older
commits or other branches. Before making a repository-wide rights-clean claim,
inspect every public ref. A coordinated history rewrite is disruptive and
requires explicit maintainer approval because it force-pushes rewritten refs
and invalidates existing clones.
