# Public repository audit

Audit date: 7 October 2026.

This document records the publication policy used for the initial Git repository. It is not a
substitute for rescanning future changes before they are pushed.

## Classification

| Classification | Paths | Reason |
| --- | --- | --- |
| **COMMIT** | `src/`, `backend/`, `frontend/`, `scripts/`, `tests/`, `config/`, `docs/` | source, tests, configuration, and documentation |
| **COMMIT** | `models/` | frozen artifacts total about 10 MB; required for inference and reproducibility |
| **COMMIT** | human-readable reports and compact research tables | evidence for the documented metrics; no runtime credentials |
| **COMMIT** | `pyproject.toml`, `README.md`, `LICENSE`, `.env.example`, `.gitignore` | packaging and public repository metadata |
| **OPTIONAL** | `data/manifest.json` | provenance, hashes, source URLs, and licences only; no source rows |
| **IGNORE** | `.venv/`, Python/test/linter caches, editor state, build outputs | machine-local or reproducible |
| **IGNORE** | `data/runtime/`, `data/product/`, `data/prospective/` | HTTP cache, SQLite runtime state, and local prospective ledger |
| **TOO LARGE** | `data/raw/`, `data/external/`, `data/processed/` | complete local `data/` tree is about 2.8 GB; largest generated file is about 542 MB |
| **SECRET / NEVER COMMIT** | `.env`, private keys, tokens, credentials, authenticated cookies | none are required by the current providers; patterns are blocked by `.gitignore` |

## Secret and personal-data scan

- No `.env` file or credential-named file was present outside the virtual environment.
- No high-confidence private-key, AWS key, GitHub token, OpenAI-style key, or Bearer-token pattern
  was found in the publishable tree.
- The only generic credential-word hit was documentation stating that no challenge cookie is
  bypassed.
- One ignored generated JSON report contained an absolute local filesystem path. JSON reports are
  excluded and the path is not part of the public index.
- Public player and tournament names in research reports are sports data, not private user data.

## Data and licence decision

The historical collections are attributed to Jeff Sackmann/Tennis Abstract and documented as
CC BY-NC-SA 4.0. They are not included. The MIT licence applies only to original code and does not
relicense third-party or derived datasets.

## Re-audit before future pushes

At minimum, inspect `git status`, run a high-confidence credential scan, and list tracked files
larger than 5 MB. Runtime SQLite files and forced additions under `data/` require explicit review.
