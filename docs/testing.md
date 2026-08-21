# Testing

`uv run pytest` runs the fast suite (≈ 440 tests, ~30 s); `-m slow` adds the long runs (6
tests), `-m orkan` the C-backend contract tests (skip without `liborkan.so`); GPU and Qiskit tests
skip without CuPy/qiskit. `sh scripts/check-all.sh` = ruff check + ruff format --check +
`mypy --strict src` + fast pytest, and is the gate before every worklog entry and commit.

## Rules

* **Red → green.** The failing test is written first and committed with the fix. Test files are
  titled "Red-phase tests for …" for that reason. A test that never failed proves nothing.
* **Channels, not marginals.** Backend and channel agreement is asserted on full distributions,
  full density matrices or Choi matrices. A `P(out=1)` check is never the only check.
* **Statistical tests state their tolerance and sample size**: ≥ 1000 samples, agreement at 4 σ
  (trajectories vs density matrix), total-variation distance at 20 000 shots (Aer vs Local).
* **Paper numbers carry their reading error** (Fig 5 bars ±0.001). A paper number we cannot
  match is a documented discrepancy in `STATE.md`, not a tuned tolerance.
* **Pins in tests too.** A test that encodes a paper fact quotes its pin (`[PTQNN Fig 5]`).
* **Tests do not write into the tree.** The compile table used to be a pytest side effect
  (review 002 #6); artefacts come from `experiments/` scripts with `meta.json`.

## Map

| file | what it pins |
|---|---|
| `test_ir.py` | unitarity/TP witnesses, `controlled`, bit order, `dagger`, no aliasing of caller arrays |
| `test_reference_backend.py` | sv ↔ dm agreement, superoperator, marginals, sampling; hypothesis property tests |
| `test_oracle_crosscheck.py` | reference vs `tests/oracle/indep_sim.py` (blind Codex implementation, brief `notes/briefs/codex-01-indep-oracle.md`) on random circuits incl. channels |
| `test_batched_backend.py` | batched kernel vs reference for any 1q/2q gate, any qubit order, `basis_columns`, `product_input_states` |
| `test_trajectories.py` | unbiasedness at 4 σ, unitary exactness with one trajectory, validation |
| `test_gpu.py` | GPU dm/trajectories vs CPU (skips without CUDA) |
| `test_orkan_ffi.py`, `test_orkan_backend.py` | struct layout, guards, `rx` convention, lsb0→msb0, all three 2q-channel routes vs reference |
| `test_shots.py` | `ShotBackend` marginals, maximal coupling moved fraction = TV, CRN variance reduction, `probe_context` scope |
| `test_neurons.py` | Fig 1/2 op sequences, closed-form `P(out=1)`, un-compute, `pack/unpack`, equal `n_params` |
| `test_train.py` | KL/cosine/ℓ2 properties, Metropolis rule, KW gradient, `minimise` bookkeeping, `restate_incumbent` |
| `test_bitcount.py` | target histogram, superposition observable, Fig 3 corners, short training converges |
| `test_iris.py` | code construction, collisions, superposition state, `fast_joint` vs circuit route, accuracy ceiling |
| `test_flappy.py` | clone determinism, baseline solves it, Fig 10 circuit/angles, fast path vs reference, policy cache, rotating-batch incumbent, margin hinge/penalty, sampled-decision flip rate |
| `test_ecg.py` | synthetic landmarks, windows/split guard band, PSNR/SSIM literal forms, batched predict vs circuit |
| `test_noise_channels.py` | CPTP of every zoo channel, Choi round trip, fidelity relations, `ReadoutError` validation; hypothesis |
| `test_noise_model.py` | ASAP schedule, channel placement order, `scaled`, Aer cross-check (skips without qiskit) |
| `test_presets.py` | every bundled TOML parses to a physical spec, readout fallback, `missing_fields`, `allow_missing` semantics |
| `test_hardware.py` (47) | every decomposition identity up to phase, gate-set synthesis per vendor, routing validity, QASM bit order, executor TVD, cost model |
| `test_*_script.py` | pure helpers of the experiment scripts (`bitcount_noise`, `flappy_noise`, `ecg_iris_noise`, `hardware_plan` — incl. the provenance validator mutation tests) |

## Adding a test

1. Put the paper pin or the invariant in the docstring; if it is our choice, say `[ASSUMPTION]`.
2. Compare full objects. For stochastic backends fix the seed *and* assert a σ-scaled tolerance.
3. Keep it under a second unless it cannot be; mark `@pytest.mark.slow` otherwise and make sure
   a fast cousin exists.
4. Run `sh scripts/check-all.sh`. Ruff ignores `E741`, `N803`, `N806`, `RUF001-003` on purpose
   (math-style names, Unicode in docstrings).

## Review rounds

Tests are one gate; independent review is the other. After a phase goes green an Opus subagent
or Codex reads the diff against `PLAN.md` and the paper pins; findings go to
`reviews/NNN-<phase>.md` with a resolution table (fixed / declined with reason). See
`reviews/README.md`.
