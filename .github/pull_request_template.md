## Summary

<!-- What changed and why? -->

## Reproducibility and data impact

- [ ] No credentials, private data, source-paper PDFs, or full-text derivatives
      are included.
- [ ] Stable identifiers and schema compatibility are preserved or migrated.
- [ ] Model/provider and generation settings are documented for regenerated
      outputs.
- [ ] Licensing and provenance are documented for new data.

## Validation

- [ ] `python -m compileall -q pipeline scripts tests`
- [ ] `python scripts/release_audit.py`
- [ ] `python -m pytest`
