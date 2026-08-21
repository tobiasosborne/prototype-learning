# tests/

`uv run pytest` (fast, ≈ 440 tests, ~30 s) · `-m slow` · `-m orkan` · GPU/qiskit tests skip when
unavailable. Rules, map of what each file pins, and how to add a test: `docs/testing.md`.

`oracle/indep_sim.py` is a deliberately direct density-matrix simulator written blind by Codex
(`notes/briefs/codex-01-indep-oracle.md`); it exists only to cross-check `ptqnn.backends.reference`
and must never import from `ptqnn`.
