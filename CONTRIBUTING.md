# Contributing to Advanced-Research

Thank you for contributing research, experiments, code, or documentation. This repository combines exploratory theoretical work with Python and Go implementations. Distinguish **proposed concepts** from **verified results** in every contribution.

## Where contributions belong

- `Ideas/`: exploratory theories, candidate mathematical frameworks, and research drafts.
- `phd/`: longer-form academic and research manuscripts.
- `src/advanced_research/`: Python implementation, including the CLI and research workflows.
- `cmd/cli/` and `pkg/`: Go implementation.
- `configs/`: sample configuration (never include credentials).
- `RESEARCH_ROADMAP.md`: ongoing and planned research work.

The roadmap also describes future `Apical-Synthesis/` and `Implementation/` directories. Those directories are not presently tracked; do not assume they exist when creating links.

## Research contribution checklist

For a new theoretical result or substantial revision, include:

1. **Research question and claim:** Define the problem and scope precisely.
2. **Assumptions and notation:** State domains, units, regularity conditions, and dependencies.
3. **Derivation or evidence:** Provide proofs where claimed, or an experimental protocol with a clear hypothesis and baseline.
4. **Reproducibility:** Record seeds, datasets or simulations, evaluation metrics, hardware, software versions, and commands.
5. **Limitations:** Identify counterexamples, negative results, uncertainty, and unverified assumptions.
6. **Provenance:** Cite sources and distinguish original contributions from AI-generated proposals or adaptations.
7. **Status:** Mark work as *proposal*, *under evaluation*, *empirically evaluated*, or *formally verified* as appropriate. Do not label a claim proven without a checkable proof.

## Code contribution checklist

- Keep changes scoped and explain the expected behavior.
- Add or update automated tests for new or changed behavior. **The repository currently lacks tracked test suites**; establishing them is an active maintenance priority.
- Avoid introducing service credentials, private datasets, or API tokens into source, examples, logs, or test fixtures.
- Keep external integrations optional and test offline behavior separately from integration tests.
- Do not modify external repositories through this repository's gitlink entries. The tracked gitlinks currently have no `.gitmodules` mapping; repair of that configuration requires confirmed target URLs.

### Python setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
# Once tests are added:
python -m pytest
```

### Go setup

```bash
go mod download
go test ./...
go vet ./...
```

The Go module currently declares Go 1.25.0 in `go.mod`. Some targets in `Makefile` refer to directories not yet tracked; check target prerequisites before using them. Test command examples above describe intended validation, not a claim that the current builds pass.

## Pull request process

1. Create a topic branch from `main`; do not commit generated artifacts, credentials, or large datasets.
2. Keep a pull request focused on one coherent change.
3. Explain motivation, relevant issue, changed files, risks, and reproduction steps.
4. Attach test output or clearly state which checks were not run and why.
5. Request review before merging changes that alter published research claims, external integration contracts, or major dependency versions.

## Licensing

The repository's [LICENSE](LICENSE) is MIT. By contributing, ensure you have rights to the included code, data, and text, and document third-party license obligations.

For questions or proposals, use the repository's issues or discussions.
