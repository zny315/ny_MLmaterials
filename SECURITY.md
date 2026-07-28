# Security policy

## Reporting

Do not open a public issue for exposed credentials, private source documents,
or other sensitive data. Use GitHub's private vulnerability reporting feature
for this repository when available, or contact the repository owner through
their GitHub profile.

Include the affected commit/path and enough detail to reproduce the issue
without pasting a live secret.

## Supported version

Security fixes target the current default branch and the latest tagged
release. Research snapshots are not guaranteed to receive backports.

## Credential handling

- Supply credentials through environment variables or an ignored local file.
- Never place keys in notebooks, logs, command history, fixtures, or model raw
  responses.
- If a key is committed, revoke it immediately; deleting the file in a later
  commit is not sufficient.
- Treat third-party model endpoints as data processors and review their
  retention and training policies before sending document content.
