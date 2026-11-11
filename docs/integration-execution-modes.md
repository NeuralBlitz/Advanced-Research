# Python integration execution modes

**Status:** Research scaffold with explicit simulation boundaries (Issue #70).

The Python modules `LRSIntegration`, `OpencodeIntegration`, and `NeuralBlitzIntegration` currently implement **simulated behavior**, not provider-backed integrations. A result produced in mock mode is not evidence that data was sent, a document was persisted, or geometry was computed.

## Configuration

Each integration accepts `enabled` (default `false`) and `mode` (default `mock`). Explicit settings for a local simulation:

```yaml
lrs:
  enabled: true
  mode: mock
opencode:
  enabled: true
  mode: mock
neuralblitz:
  enabled: true
  mode: mock
```

With `mode: real`, enabled operations raise `IntegrationNotImplementedError` because these Python provider adapters are not implemented. This fail-closed behavior is intentional. Do not enable `real` expecting automatic connectivity. A separate Go integration layer exists but is not a substitute for Python live-provider validation.

## Mock result semantics

- **LRS:** records enter a local in-memory queue; `flush_records` does **not** transmit or clear that queue. Console output reports **0 sent**. The queue is lost on process exit.
- **Opencode:** document creation returns identifiers prefixed `mock:`; nothing is saved to a provider or durable file. Search returns fixed results marked `mock: true`.
- **NeuralBlitz:** fixed example eigenvalues, curvature data, and optimization statistics are returned with `mock: true` and `status: simulated`. They are not derived from the supplied input, and `computation_time: null` is not a benchmark.

The Python `test-integrations` CLI command reports modes and availability, not external connectivity. Use `PYTHONPATH=src python -m pytest -q tests/test_integration_modes.py` to exercise the simulation boundary without network access.

## Requirements before a real provider may be enabled

A real adapter must implement authentication/configuration, actual side effects, explicit error handling, idempotency or duplicate protection where applicable, and integration tests against a controlled provider or test server. For numerical computation, demonstrate input dependence, analytical baselines, and numerical tolerances before claiming measured geometric output. Preserve mock mode for deterministic regression testing, but never silently fall back from real mode to mock mode.

**Repository scope:** Only `NeuralBlitz/Advanced-Research`. Do not mutate third-party repositories or services as part of these tests.
