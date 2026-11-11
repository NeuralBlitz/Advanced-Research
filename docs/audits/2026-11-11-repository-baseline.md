# Advanced-Research: repository baseline audit

**Inspection date:** 2026-11-11  
**Scope:** `NeuralBlitz/Advanced-Research` only  
**Baseline:** `main` at `9a8235ece79944e431af0bfca44142814e48f23c`  
**Method:** Read-only inspection of GitHub repository metadata, recursive Git tree, selected source files, configuration, and pull-request metadata. No full install, build, runtime test, or scientific reproduction was performed.

## Inventory

| Item | Observed |
| --- | ---: |
| Tracked Git tree entries (including directories) | 357 |
| Regular Git blobs | 283 |
| Directories in tree | 46 |
| Gitlinks (`160000` mode) | 28 |
| Python source files | 11 |
| Go source files | 9 |
| Markdown files | 251 |
| Tracked Python tests under `tests/` | 0 |
| Tracked Go `_test.go` files | 0 |
| Branches returned by the GitHub branches endpoint | 22 |
| Open pull requests | 4 |
| Open non-PR issues | 0 |

## Verified findings and proposed remedies

### P0 — repository integrity and maintenance

1. **Missing gitlink mapping:** There are 28 gitlinks in the Git tree, but no tracked `.gitmodules` file. Normal recursive submodule initialization cannot resolve their URLs from this commit. **Unresolved.** Recover each intended URL and commit mapping before writing a `.gitmodules` file; do not guess or repoint any gitlink.
2. **Dependabot ecosystem invalid:** `.github/dependabot.yml` has a single `package-ecosystem: "pip, npm"` entry. GitHub requires a single supported ecosystem per entry. **Proposed fix in this audit branch:** separate valid `pip` and `gomod` weekly entries, matching `pyproject.toml`/`requirements.txt` and `go.mod`.
3. **Unprotected default branch:** The GitHub branch endpoint reports `main` as not protected. **Unresolved.** Define meaningful required checks before changing branch rules; avoid merging without review and verification.

### P1 — engineering and reproducibility

4. **No tracked automated test suite:** The recursive tree contains no `tests/` suite or Go `_test.go` files, and no `.github/workflows/` directory exists. GitHub Actions history includes scheduled activity but does not establish working test CI for this code. **Unresolved.** Add deterministic unit tests and Python/Go build-and-test workflows.
5. **Go CLI import-name mismatch:** `pkg/core/context.go` declares `package context`, while `cmd/cli/main.go` imports its path without an alias and refers to `core.DefaultConfig` and `core.NewContextInjector`. The imported identifier would be `context`, not `core`. **Unresolved.** Fix imports and verify compilation before enabling Go CI.
6. **Go targets and documentation out of sync:** `Makefile` and `README_GO.md` refer to `cmd/server`, `cmd/worker`, `api/`, and `internal/` paths absent from the observed tree. The Go README also points to a different clone path. **Unresolved.** Implement those targets or clearly mark them as planned; keep `make build` and Go CLI docs accurate.
7. **Read the Docs configuration not backed by source:** `.readthedocs.yaml` references `docs/conf.py`, which is absent from the observed tree. **Unresolved.** Provide a valid Sphinx project or change the documentation build approach.
8. **No committed `go.sum`:** `go.mod` exists, but no `go.sum` is tracked in the recursive tree. **Unresolved.** Generate and commit dependency checksums after a clean, reproducible module resolution.

### P2 — consistency and research governance

9. **Dependency duplication:** `pyproject.toml` and `requirements.txt` repeat several package specifications, including `asyncio-throttle`, `markdown`, `pytest-asyncio`, and `black`. There are overlapping but differing lower bounds. **Unresolved.** Consolidate constraints while coordinating with open Dependabot PRs #56–#59.
10. **Conflicting advertised license:** `LICENSE`, `README_GO.md`, and `pyproject.toml` specify MIT; the primary README advertised Apache 2.0. **Proposed fix in this audit branch:** align the README with the actual MIT LICENSE; the license itself is unchanged.
11. **Contribution guide empty:** `CONTRIBUTING.md` contains only a dot. **Proposed fix in this audit branch:** standards for evidence, provenance, reproduction, testing, and reviews.
12. **Research roadmap ahead of implementation:** `RESEARCH_ROADMAP.md` references `Apical-Synthesis/` and `Implementation/`, neither present in the tracked tree. **Unresolved.** Clarify status and add traceable theory-to-experiment indexing before claiming implementation completion.

## Scientific validation policy

A manuscript's presence in `Ideas/` or `phd/` establishes that it is documented, **not** that its mathematical statements are proven, its benchmarks reproduced, or its applications independently validated.

A subsequent research review should assess claims at three distinct layers:

1. **Formal:** notation, assumptions, proofs, counterexamples.
2. **Computational:** executable implementation, tests, reproducibility, baselines.
3. **Empirical:** datasets, metrics, uncertainty, independent replication.

Maintain traceability from a theory document to its experiment protocol, code revision, result artifact, and independent review.

## Proposed next implementation sequence

1. Confirm submodule targets and repair `.gitmodules` without moving existing gitlink SHAs.
2. Establish a reproducible Python and Go test baseline, beginning with Go compile errors and Python CLI smoke tests.
3. Add the smallest working GitHub Actions checks, then set branch protection to require them.
4. Repair documentation generation and broken Makefile targets.
5. Consolidate dependency manifests while reviewing Dependabot PRs.
6. Build a research claim-and-evidence index for the 251 Markdown files.

## Verification boundary

This report is based on retrieved repository state at the stated baseline; it is **not** a statement that builds, tests, documentation publication, or experimental claims passed. Changes proposed in the audit branch must be reviewed and merged before they affect `main`.

## Post-baseline resolution update — 2026-11-11

The measurements and findings above describe `main` at the original inspected SHA `9a8235ece79944e431af0bfca44142814e48f23c`. They should be preserved as a historical baseline, not read as a claim about the present branch after subsequent merges.

- [PR #61](https://github.com/NeuralBlitz/Advanced-Research/pull/61) was **merged** as `480a55e6ca69b5171ebe84c00828762ba85b0387`. It established core Python/Go regression tests and a passing CI workflow, repaired the observed Go compiler blockers, and committed a resolved `go.sum`. This partially addresses findings 4 and 5 and closes the missing-checksum concern in finding 8.
- Python/Go runtime service integrations, production server behavior, and independent mathematical validation were not established by those checks.
- The 28 Gitlink URL mappings, missing `.gitmodules`, Read the Docs configuration, dependency duplication, and roadmap organization remain tracked follow-ups.

