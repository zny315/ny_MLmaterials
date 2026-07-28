# Contributing

Thank you for improving this research artifact.

## Before opening a change

1. Open an issue for large schema, dataset, or behavioral changes.
2. Do not commit source-paper PDFs, full-text derivatives, API responses that
   contain copyrighted text, credentials, or machine-specific paths.
3. Keep generated outputs under ignored `outputs/` paths.
4. Preserve stable identifiers and document schema migrations.
5. Add or update tests for behavior changes.

## Development setup

```bash
python -m venv .venv
python -m pip install -r requirements-dev.txt
```

Run the offline checks:

```bash
python -m compileall -q pipeline scripts tests
python scripts/release_audit.py
python -m pytest
```

## Pull requests

Keep pull requests focused. Describe:

- the motivation and affected pipeline stage;
- any schema or data migration;
- the exact checks run;
- model/provider details when outputs were regenerated;
- licensing or provenance implications for new data.

Never include real API credentials in issues, logs, fixtures, or screenshots.
By contributing code, you agree that it is provided under the repository's
MIT license. By contributing authored structured data, you agree that it is
provided under CC BY 4.0.
