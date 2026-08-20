# Review 002 — noise layer, fast backends, tasks, hardware layer (P4–P10)

Reviewer: independent Opus subagent. Date: 2026-08-20.
Commit under review: `871ee9e` ("docs: Iris results").

**Moving-target caveat.** The main session kept committing while this review ran; by the time it
was written HEAD had advanced to `a330088` (ECG, a density-matrix perf refactor `30564d9`, and
fixes from `reviews/002b-codex-noise-adversarial.md`).  Findings below are against `871ee9e`
unless stated otherwise; a clean export of that commit lives at `/tmp/ptqnn-head` and was used
wherever the two differ.  Two findings (2, and part of 13) concern code that landed *after*
`871ee9e` and are flagged as such — they are reported because the brief asks for regressions.

Scope: `src/ptqnn/noise/{channels,model,presets}.py` + `devices/*.toml`;
`src/ptqnn/backends/{trajectories,gpu,orkan}.py`; `src/ptqnn/tasks/{flappy,iris}/*`;
`src/ptqnn/hardware/*`; `src/ptqnn/train/kwsa.py` (`restate_incumbent`, `map_fn`);
`experiments/{bitcount_figs,flappy_train,iris_fig7_fig8_fig9}.py`; matching tests;
`reports/hardware-plan.md`, `reports/reproduction.md`.

Method: executed, not read off.  `sh scripts/check-all.sh` (green, 354 passed, 6 deselected),
`uv run pytest -m orkan`, GPU runs with `LD_LIBRARY_PATH=/usr/lib/wsl/lib`, and throwaway
scripts under `/tmp` (`chk_noise1`, `chk_idle2/4/5`, `chk_ro`, `chk_dev`, `chk_dur`, `chk_hw`,
`chk_qasm`, `chk_traj`, `chk_gpu`, `chk_iris`, `chk_misc`, `chk_kwsa`).  No repository file was
modified except this one.

---

## What checks out (tried hard to break, could not)

* **`NoiseModel` idle scheduling is exactly Qiskit's ASAP scheduler.**  On a deliberately
  unbalanced 3-qubit circuit (`rx(0)×3, crx(0,1), rx(2), crx(1,2), rx(0)`, 50/300 ns),
  `NoiseModel.apply` inserts idle channels of 150 ns on q1, 400 ns on q2 and a 250 ns final pad
  on q0.  Qiskit 2.5's own `ASAPScheduleAnalysis` + `PadDelay` against a `Target` with the same
  durations emits `delay 150` on q1, `delay 400` on q2, `delay 250` on q0 — identical, including
  the final padding (`/tmp/chk_idle4.py`).
* **Aer cross-check *with* idling, which the repo's own test does not do.**  Same circuit, with
  `depolarizing(0.01/0.03)`, `T1=80 µs`, `T2=60 µs`, and the Aer side built from labelled
  identity unitaries carrying `thermal_relaxation_error` for the three idle windows:
  `max|p_ours − p_aer| = 1.67e-15` over the full 8-outcome distribution (`/tmp/chk_idle5.py`).
  The scheduling *and* the channel algebra are right.
* **Depolarising p ↔ fidelity.**  `ch.depolarizing(p, n)` and Aer's `depolarizing_error(p, n)`
  have identical Choi matrices (≤ 4.3e-19) for `n ∈ {1,2}`, `p ∈ {0.01, 0.1, 0.5}`;
  `average_gate_fidelity` agrees with `qiskit.quantum_info.average_gate_fidelity` to 8 digits;
  `depolarizing_probability(1−ε, n)` round-trips (`ε = 3e-3 → p = 6e-3` (1q) / `4e-3` (2q), both
  back to `1 − F_avg = 3e-3`).
* **Thermal relaxation, including `T2 > T1`.**  Choi matrices agree with Aer's
  `thermal_relaxation_error` to ≤ 6.7e-16 for `(T1,T2)` = (80,60), (80,120), (80,160), (50,90)
  with `excited_state_population = 0.02`, (100,100) and (30,15) with `n_eq = 0.05`.  The Kraus
  rank drops to 2 exactly at `T2 = 2T1`, as it must.
* **Readout confusion orientation.**  `ReadoutError(p01, p10)` builds `M[read, true]`; on
  `|q0=0, q1=1⟩` with ibm_fez's `p01=3.906e-3, p10=1.147e-2` it gives
  `(0.011430, 0.984664, 4.48e-5, 0.003861)`, matching Aer's `ReadoutError([[1−p01,p01],[p10,1−p10]])`
  at 4×10⁶ shots to within Monte-Carlo error (`/tmp/chk_ro.py`).  `ibm_fez.toml:75-76` defines
  `p01 = prob_meas1_prep0`, `p10 = prob_meas0_prep1` — the same orientation as the code, and the
  right one physically (`p10 > p01`: |1⟩ decays during the 1560 ns readout).
* **Trajectory branch sampling is correct at the edges.**  `(u >= cdf).sum()` is the inverse-CDF
  rule; a zero-norm branch has `cdf[i] == cdf[i−1]` and is provably unreachable, so the
  `br[sel]/√norms[sel,i]` division never sees 0.  Verified on `amplitude_damping(1.0)` applied to
  |0⟩ and |1⟩, and on a Kraus set containing a literal zero operator: no NaN, exact answers.
  Agreement with the exact density matrix on a 5-qubit noisy circuit: `max deviation = 1.06 σ`
  at N=2000 and `0.59 σ` at N=20000 (`/tmp/chk_traj.py`).
* **GPU dispatch (at HEAD).**  `GpuDensityMatrixBackend` matches the CPU reference to `0.0` on a
  5-qubit noisy circuit and on an `initial_state` circuit; `GpuTrajectoryBackend` agrees at
  1.43 σ with N=4096, handles zero-probability branches, and the unitary shortcut works
  (`/tmp/chk_gpu.py`, run against `/tmp/ptqnn-head`).  CuPy really does flow through
  `np.tensordot/moveaxis/cumsum/stack/empty_like` unchanged.
* **orkan adapter.**  `depolarizing_strength` verifies the *whole* Choi matrix rather than just
  the fidelity, so a same-fidelity non-depolarising channel cannot be mistaken for one; the
  factored twirl `√q·{P/2}` on the first qubit and `{P/2}` on the rest reproduces
  `(1−q)ρ + q I/d·Tr ρ` exactly; `from_numpy`/`to_numpy` use the bit-reversal permutation, which
  is an involution, so the round trip is consistent.  `-m orkan` is green.
* **Transpiler.**  Exhaustively checked (`/tmp/chk_hw.py`): `decompose_crx` is *exact* (not just
  up to phase) for θ ∈ {0, 0.7, π, −2.1}; `_synth_1q` reproduces 17 unitaries — including the
  degenerate cases I, X, Z, H, S, RZ(π), RX(π), RY(π), RX(1e-9), RY(π−1e-9) and 6 Haar-random
  ones — up to global phase in *every* gate set, and emits only native names in every case;
  full-network `to_gateset` is phase-equivalent in all six sets; a 2q gate on a non-adjacent pair
  with `control > target` is handled correctly.  The `zyz_angles` degeneracies at `b = 0` and
  `b = π` (where `angle(0)` is taken as 0) are harmless because only `a+c` resp. `a−c` is
  observable there — I checked this specifically.
* **Routing correctness.**  `U_routed = Q_final · U_ideal · Q_initial†` holds to 1e-10 on linear,
  ring and heavy-hex maps with 4 SWAPs inserted.
* **QASM3 and counts bit order.**  Round-tripping every gate set through
  `qiskit.qasm3.loads` and conjugating by `bit_reversal_permutation` reproduces the IR unitary up
  to phase in all six sets.  `Operator(to_qiskit(c)).data` equals the IR unitary *directly*.
  `LocalExecutor` and `QiskitAerExecutor` both return keys in `BITSTRING_ORDER`: TV to the exact
  distribution 0.0016 and 0.0026 at 2×10⁵ shots.  `counts_from_qasm({'001'}) = {'100'}` for
  `X` on qubit 0 — correct.
* **Iris read-out claims reproduce.**  On `runs/iris/20260820T145455Z` (seed 2) the prototype net
  gives test accuracy 0.9800, input-register leak 0.0011, and the *per-sample basis* read-out
  gives the same 0.9800 — exactly the argument `tasks/iris/model.py:28-35` makes.  Classic:
  0.9100 / leak 0.0000 / 0.9100.
* **Flappy encoding default.**  `ANGLE_MAPS["amplitude"] = 2·asin(√x)` is the default everywhere
  (`agent.py:48-58`, `ObjectiveConfig.angle_map`), pinned by `[SNN Fig 4]`'s printed
  `CRX(1.369) = 2·asin(√0.4)`; the flappy env is bit-identical under a fixed seed.
* Every vendor number I spot-checked in `ibm_fez.toml`, `quantinuum_h*.toml` and
  `ionq_forte.toml` carries a `[[provenance]]` table with URL, `retrieved_at` and a verbatim
  quote, and the TOML's own internal cross-checks (median-of-means ≠ mean-of-medians, the
  1.5× CZ drift, the 2024 vs 2025 epoch) are honest and unusual in their care.

---

## Findings

Severity is stated on each heading; the numbering is not a severity order (19 is a major that
landed last because the mutation run finished last).  A suggested fix order is in the verdict.

### 1. `reports/hardware-plan.md` is not backed by run dirs and contains a dozen wrong numbers — blocker

`reports/hardware-plan.md` (whole file).  The report cites **no run dir anywhere**, in a repo
whose stated convention is "Reports in `reports/` cite run dirs" (`CLAUDE.md`).  Worse, numbers
that *can* be checked are wrong.  Verified individually:

| line | claim | what the repo says |
|---|---|---|
| 317 | "`experiments/runs/bitcount_noise/20260820T145335Z` exists but is empty" | it holds `results.json`, `results.md` and four figures |
| 323–324 | "TVD 0.035 (H2-1: 0.034)" for the 11 q bit-count | `results.json A/prototype/per_device`: H1-1 `tv_clean_vs_noisy = 0.00626`, H2-1 `0.00486`; even `tv_target` is 0.0154/0.0148 — 2.3–7× smaller |
| 324 | "`P(out=1)` moves by only 0.0006 (0.4813 → 0.4807)" | the run's `kept_mean` is **0.48709**, and §1.3(b) of the same report says 0.4871.  0.4813 appears nowhere |
| 192 | "H2-1 … the lowest 2q error of any preset (1.1e-3)" | `quantinuum_h1_1.toml:` `error_median = 9.7e-04` — H1-1 is lower |
| 226 | "`2^{n+1} − 2n − 2 = 4 070` CNOTs" | for n=12 that formula is **8166**; 4070 is `2^n − 2n − 2` |
| 307 | "67 % (Flappy) / 73 % (bit-count) of entanglers are SWAP" | from the compile table: Flappy `3·22/90 = 73.3 %`, bit-count `3·89/327 = 81.7 %` |
| 312–314 | "Garnet's T2-echo (21 µs) is shorter than our routed bit-count critical path (18 µs)" | 21 > 18; the sentence is backwards (both numbers are individually right) |
| 156 | "45 ms QPU ⇒ ≈ $3–8" | `0.045 × $1.60 = $0.07`; the repo's own `cost_estimate` returns $48.07 because of a 30 s job overhead.  Neither endpoint is derivable |

**Fix**: regenerate the report from a script under `experiments/` that writes
`experiments/runs/hardware-plan/<stamp>/` with `meta.json`, and quote only numbers that script
emits.  Where a number is an estimate, say which formula and which input produced it.  The
fidelity arithmetic in §2.2, the SABRE table in §3.2 and the cost table in §2.3 *do* reproduce
exactly, so the report is salvageable — but as it stands a reader cannot tell the checked rows
from the unchecked ones.

### 2. The GPU density-matrix backend is broken at HEAD, and `check-all.sh` was red when it was committed — blocker

`src/ptqnn/backends/reference.py:57-63`, introduced by `30564d9` ("perf: density-matrix ops via
block superoperator"), which landed while this review was running and is now **committed**
(confirmed still failing at `a330088`):

```python
s: Complex = np.zeros((d2, d2), dtype=np.complex128)
for k in kraus:
    s = s + np.kron(k, k.conj())
```

The accumulator is seeded with a **numpy** array, so when `kraus` holds CuPy arrays
`np.kron(cupy, cupy)` is added to a numpy array and CuPy raises
`TypeError: Unsupported type <class 'numpy.ndarray'>`.  Evidence:

```
$ LD_LIBRARY_PATH=/usr/lib/wsl/lib uv run pytest tests/test_gpu.py -q
FAILED tests/test_gpu.py::test_gpu_density_matrix_matches_cpu   - TypeError
FAILED tests/test_gpu.py::test_gpu_unitary_and_initial_state    - TypeError
```

against `/tmp/ptqnn-head` (commit `871ee9e`, i.e. before `30564d9`) the same script prints
`DM max|diff| = 0.0`.  The commit message quotes a GPU speed-up ("GPU 18 s → 1.5 s") for a code
path that no longer runs at all.

`scripts/check-all.sh` **does** catch it on this machine — `gpu_available()` returns `True` in a
plain shell here (`LD_LIBRARY_PATH` is *not* needed today, contrary to the `CLAUDE.md` gotcha),
so `tests/test_gpu.py` runs:

```
$ sh scripts/check-all.sh
FAILED tests/test_gpu.py::test_gpu_density_matrix_matches_cpu
FAILED tests/test_gpu.py::test_gpu_unitary_and_initial_state
2 failed, 382 passed, 6 deselected in 275.91s
```

So this is a **process** failure, not a tooling gap: `30564d9` and the three commits after it were
made with `check-all.sh` red, against `CLAUDE.md` §"Session close" item 1 ("`sh
scripts/check-all.sh` green (or the failure recorded verbatim)").  Nothing in `STATE.md` or the
worklog records the failure.

A secondary robustness point stands: `tests/test_gpu.py:15` is a module-level `skipif` on
`gpu.gpu_available()`, and pytest's default `-q` summary shows skips only as a count, so on a
machine where CuPy *does* need `LD_LIBRARY_PATH=/usr/lib/wsl/lib` the whole file would vanish
without anyone noticing.

**Fix**: (a) seed the accumulator from the data —
`s = sum((np.kron(k, k.conj()) for k in kraus[1:]), np.kron(kraus[0], kraus[0].conj()))`;
(b) add `LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/lib/wsl/lib` to `scripts/check-all.sh` and run it
with `-ra` so skipped files are named in the summary; (c) the mutation audit (finding 19) shows
the GPU tests are the weakest in the tree — two of the six surviving mutations are in
`gpu.py` — so fix those at the same time.

### 3. `route()` and `compile_for()` silently discard `Circuit.initial_state` — major

`src/ptqnn/hardware/transpile.py:551` — `out = ir.Circuit(n_phys, initial_state=None)`.

Evidence (`/tmp/chk_hw.py`):

```
original initial_state is None? False   compiled? True
```

An Iris circuit carries the whole dataset superposition in `initial_state`
(`tasks/iris/model.py:233`).  `compile_for(iris_circuit, device)` returns a circuit that starts
from |0…0⟩ and reports metrics for a *different program*, with no warning.  `to_gateset` does
preserve `initial_state`, and `to_openqasm3` correctly *raises* on it — so the router is the odd
one out, and it is the one that silently loses information.  §3.3 of the hardware plan reasons
about exactly this state, which makes the silent drop dangerous rather than academic.

**Fix**: `route` should raise the same `ValueError` `to_openqasm3` raises
("synthesise the state preparation into gates first"), or carry the state through and permute it
by `permutation_unitary(pos, n)`.  Add a test asserting the raise.

### 4. `readout.duration_ns` is dead in both the noise model and the cost model; in the cost model that is a ~2.5× undercount — major

`src/ptqnn/noise/presets.py:250, 301` parse `readout_ns` from the TOML, with its own
`[[provenance]]` entry (`ibm_fez.toml:192-197`, 1560 ns).  Nothing ever reads it:
`DeviceSpec.noise_model()` (`presets.py:254-264`) does not pass it on, `NoiseModel` has no
measurement duration, and `transpile.metrics` (`transpile.py:336-380`) walks `circuit.ops` only,
so `critical_path_ns` stops at the last gate.

**In the noise model the omission is defensible and should be *said*.**  I first scored this as
a missing relaxation channel — adding one idle of 1560 ns per qubit before the readout map moves
a 7-qubit prototype network's distribution by `TV = 0.024`, larger than any device's total
clean→noisy TV in the noise study — and then found that it would **double count**:

```
1 − exp(−1560 ns / 144.855 µs) = 0.010712
ibm_fez p10 (prob_meas0_prep1 median) = 0.0114746      ratio 1.07
ibm_fez p01 (prob_meas1_prep0 median) = 0.0039         (no decay mechanism, as expected)
```

IBM's asymmetric `p10` *is* the decay during readout — `ibm_fez.toml:76-77` says so in words
("|1> decays toward |0> during the 1560 ns readout") — and since measurement is diagonal, a
classical confusion map is a complete description of it.  The code is right.  But nothing in
`presets.py` or `model.py` records that reasoning, so the next person to notice the unused field
will "fix" it and silently double the modelled readout error.  **Fix**: state it in the
`[ASSUMPTION]` block at `presets.py:213-218`, with the arithmetic above.

**In the cost model it is a real error.**  `_wall_clock_s` (`executor.py:466-469`) multiplies
shots by `critical_path_ns`, which for the Fig-5 network on ibm_fez is 1056 ns — while the
measurement that follows every shot takes 1560 ns, i.e. **more than the whole circuit**.  Every
`ibm_pay_as_you_go` figure, and the "45 ms QPU" line of `reports/hardware-plan.md:156`, is short
by ~2.5×.  **Fix**: add `readout_ns` (and reset, where published) to the duration map
`compile_for` builds, or add it explicitly in `_wall_clock_s`; note in `CompiledCircuit.notes`
when the vendor publishes no readout duration (Willow, IonQ, IQM, Rigetti all omit it).

**A third, smaller point.**  For every preset except ibm_fez, `parse_device` sets
`p01 = p10 = error_median` (`presets.py:299-300`), which by construction cannot represent
decay-during-readout — the asymmetry is a *physical* signature, not a rounding.  That is a
faithful transcription of what those vendors publish, but it is invisible at the call site; see
finding 5 for the general shape of the problem.

### 5. `parse_device` turns "not published" into "perfect" — major

`src/ptqnn/noise/presets.py:295-301`: `g1.get("error_median", 0.0)`, `ro.get("error_median", 0.0)`.
Contrast `presets.py:282-284`, which correctly *drops* relaxation when only one of T1/T2 is
known rather than guessing.

Consequence (`/tmp/chk_dev.py`):

```
rigetti_ankaa_3  T1=None T2=None e1=0.000e+00@0.0ns e2=1.000e-02@72.0ns ro=(0.0,0.0)
```

`rigetti_ankaa_3.toml:5` says outright that "fidelity, readout error and readout duration are
NOT public"; the loader turns that into a device with *perfect* single-qubit gates and *perfect*
readout.  In the noise study it then scores `tv_clean_vs_noisy = 0.0293` — better than ibm_fez's
0.0285 would have been with an honest 1q error, and any cross-device ranking that includes it is
meaningless.  `iqm_garnet` has the same problem in weaker form: `gate_1q.duration_ns = 0`, so no
relaxation is charged during 1q gates at all.

**Fix**: make the missing fields `None` in `DeviceSpec`, and have `noise_model()` either omit the
corresponding channel *and record it in `NoiseModel.name`/a `missing` tuple*, or raise unless the
caller passes `allow_missing=True`.  At minimum, `DeviceSpec` should expose
`missing_fields: tuple[str, ...]` so `reports/noise.md` can print it next to every number.

### 6. A test writes a tracked file into `reports/` — major

`tests/test_hardware.py:602-627`, `test_compile_table_for_flappy_and_bitcount`, generates
`reports/hardware-compile-table.md`.  Three problems, all structural:

* a test mutates the repository as a side effect — `uv run pytest` is no longer safe to run on a
  clean tree, and a `git diff` after a test run is noise;
* the artefact lands in `reports/` instead of `experiments/runs/<name>/`, so it has **no**
  `meta.json`, no seed, no git hash and no backend — the exact three fields `CLAUDE.md` requires;
* the file is simultaneously an input (`experiments/bitcount_noise.py:1184` reads it) and a test
  output, so the noise study's 2q-gate counts depend on whether someone last ran pytest.

The table's contents are correct — I re-derived all 36 rows and the identity
`n_2q = 2·n_CRX + 3·n_swap` holds — this is purely about where it comes from.  One consequence
of the missing `meta.json` is concrete: the `n_1q` column depends on the *parameter vector*
(`_synth_1q` drops rotations that are `0 mod 2π`), and the generator's `default_rng(17)` is
recorded nowhere.  Random vectors are stable (n_1q = 2007–2008 over five seeds for the
ibm_heron/heavy-hex bit-count row) but `np.zeros(n_params)` gives 132, so the numbers are only
reproducible if the seed travels with them.

**Fix**: move the generator to `experiments/hardware_compile_table.py`, write to
`experiments/runs/hardware/<stamp>/` with a `meta.json`, and leave in `tests/` only an assertion
on a handful of the numbers (e.g. the all-to-all rows have `n_swap = 0`, heavy-hex has
`n_2q = 2·n_CRX + 3·n_swap`).

### 7. Three private hand-rolled simulators have grown because the `Backend` protocol is too narrow — major (design)

Review 001 finding #5 asked for a wider `Backend` and was declined ("every planned experiment is
≤ 17 qubits").  The cost has now materialised as *three* independent statevector kernels that
bypass `Backend` entirely:

* `tasks/flappy/agent.py:124-149` — `_input_basis_columns` + `_fast_activation` (evolve the
  `2^n_in` input basis once, then one matvec per observation);
* `tasks/iris/model.py:114-189` — `_apply_2x2`, `_apply_gate`, `output_activation_table`,
  `fast_joint` (an in-place `rx`/`crx` kernel plus an analytic contraction of the output qubit);
* `tasks/ecg/model.py:101-140` — `_apply_gate_batched`, `_batched_activation` (out of this
  review's scope, but the same pattern a third time).

Each is `rx`/`crx`-only (`iris/model.py:136` raises on anything else), each is exact for
*noiseless pure states only*, and each is selected by a different ad-hoc rule
(`isinstance(self.backend, ref.StatevectorBackend)` in flappy, `backend is None` in iris).  The
consequences are already visible: there is no fast path for a *noisy* Flappy or Iris run, so P9
for those tasks has to fall back to the slow `Backend` route, and any future gate (a native
`prx`, a `swap` from routing) silently breaks two of the three.  Alongside this sit three
`kl_loss`, two `evaluate` and three copies of `git_hash()`.

**Fix**: widen `Backend` as review 001 proposed — `probabilities(circuit, qubits=None)` plus an
optional `batch(circuits) -> array` — and put the two tricks that actually matter behind it: a
`LinearInputBackend` (flappy's basis-column trick, valid for any prep-then-fixed-unitary circuit)
and the output-qubit contraction (a `Network`-level method, not a task-level one).  Then
`fast_joint`, `_fast_activation` and `_batched_activation` collapse into one implementation with
one cross-check against the oracle.

### 8. `restate_incumbent` landed but is not wired, and the docs still describe the old limitation — major

`src/ptqnn/train/kwsa.py:59-63` added `restate_incumbent` precisely because of Flappy's rotating
seed batch.  But:

* `src/ptqnn/tasks/flappy/train.py:20-24` still says a rotating batch "would need an optimiser
  that re-scores the incumbent" — it now exists;
* `src/ptqnn/tasks/flappy/train.py:131-141` (`SeedRotation`) repeats the claim;
* `reports/reproduction.md:79-80` records it as a *negative result*: "rotating training seeds per
  iteration breaks KW–SA because the incumbent loss is not re-scored (`kwsa.minimise` is built
  for stationary objectives)";
* and `train()` never sets it: `tr.DEFAULT_KWSA.restate_incumbent = False` (verified), including
  on the `batch > 0` path (`train.py:244-249`) which is the one use case the flag was written for.

So the repo simultaneously ships the fix, does not use it, and reports the unfixed behaviour as a
finding.  The "0/25 perfect games" number that justifies `batch=0` is now untested.

**Fix**: set `restate_incumbent=True` automatically when `train()` is given `batch > 0` (or
refuse the combination), re-run the 25-seed experiment, and update both the module docstring and
`reports/reproduction.md` with whichever way it comes out.

Related, and worth deciding at the same time: with `restate_incumbent=True` the returned
`KWSAResult.params/loss` are still *best ever seen*, which under a non-stationary objective
selects the point that drew the luckiest batch, and `loss` is a number from one particular batch
that is not comparable across runs (`/tmp/chk_kwsa.py` returns `loss = −4.65` for a point whose
stationary value is 0).  For non-stationary objectives `final_params` is the honest answer.

### 9. The Iris split assumption is justified by a Fig-8 bin count the runs contradict — major (paper fidelity)

`reports/reproduction.md:67` — "Assumptions: 1/3 stratified train split (**matches Fig 8 bin
counts 15/30**)"; `src/ptqnn/tasks/iris/data.py:19-23` makes the same argument.

The three Iris runs record:

```
20260820T144849Z seed 0  distinct_train_codes 14  distinct_test_codes 20
20260820T145454Z seed 1  distinct_train_codes 12  distinct_test_codes 20
20260820T145455Z seed 2  distinct_train_codes 14  distinct_test_codes 18
```

The train side (12–14 vs 15) is a defensible near-match; the **test side is 18–20, not 30**, and
that half of the argument simply does not hold.  Since the split ratio is the load-bearing
`[ASSUMPTION]` behind the 95/91/98 % accuracy row, the justification needs to be either repaired
(what `train_size` actually produces ~30 distinct test codes? a 1/5 split?) or withdrawn and
replaced by "we chose 1/3; here is the sensitivity".

Two smaller companions in the same table: "encoding ceiling 97 % on our split" is true for seed 0
only (the metas give 0.97 / **0.99** / **0.99**), and "test bottoms at 0.033" is seed 0 only
(seed 1 bottoms at 0.0242 @ sweep 4, seed 2 at 0.0350 @ sweep 2) although the row is headed
"seeds 0/1/2".

Separately, the `(label+1)²` target itself I could not falsify and find well argued: the bar
heights `1/86, 4/86, 9/86` with bin counts (3,5,7) and `1/154, 4/154, 9/154` with (7,12,11) are
two independent integer normalisations, and no other exponent produces them (exponent 1 would
give `1/34, 2/34, 3/34`).  It is a figure read-off that the extracted text cannot confirm — the
paper says only "the amplitude encodes the class label" — so it deserves the explicit reading
error `CLAUDE.md` asks for, but the reasoning is sound.

### 10. `cost_estimate`'s Braket price table is unreachable for every shipped device, and the block is duplicated — major

`src/ptqnn/hardware/executor.py:322-337` and again at `:406-418` — two byte-identical Braket
branches, both gated on `("aws" in vendor or "braket" in vendor)`.  No shipped device TOML has
`vendor` containing "aws" or "braket" (they are `IonQ`, `IQM`, `Rigetti`, `IBM`, …), so:

```
iqm_garnet       model=wall_clock          30.0156 seconds
rigetti_ankaa_3  model=wall_clock          30.0375 seconds
google_willow    model=wall_clock          30.0573 seconds
```

`_BRAKET_PER_SHOT["iqm_garnet"]` and `["rigetti_cepheus"]` are dead entries; the second branch is
dead code outright.  Anything in `reports/hardware-plan.md` §2.3 that quotes a Braket dollar
figure for Garnet was therefore computed by hand, not by this function.

**Fix**: delete the duplicate block; key the lookup on the device slug alone (or add a
`braket_arn`/`access` field to the TOMLs and gate on that); add a test that every device in
`available_devices()` produces a `CostEstimate` whose `model` is not `wall_clock` unless the TOML
says the device has no published price.

### 11. `gateset_for` silently mislabels two of the seven presets — major

`src/ptqnn/hardware/transpile.py:684-701`.  Measured on the Fig-5 network:

* `google_willow` (`native_1q = ["xy_microwave", "virtual_z"]`, `native_2q = ["cz"]`) falls
  through to `return GATESETS["ibm_heron"]` and is compiled into `{rz, sx, x, cz}` — a basis
  Google does not have.  `CompiledCircuit.gateset.name == "ibm_heron"` while
  `device_name == "google_willow"`.
* `rigetti_ankaa_3` (`native_2q = ["iswap"]`) is compiled into CZ, because
  `RIGETTI_ANKAA.two_qubit[0] == "cz"`.  This *is* documented — in `gatesets.py:106-107` — but
  nothing propagates the caveat to `CompiledCircuit.notes`, which is what a report reads.

**Fix**: `gateset_for` should raise (or return a `GateSet` plus an explicit
`"[ASSUMPTION] no native set for <device>; using <gs>"` note) instead of defaulting to IBM, and
`compile_for` should append that note and the iSWAP caveat to `CompiledCircuit.notes`.

### 12. The `NoiseModel` docstring misattributes its composition order to Aer — minor

`src/ptqnn/noise/model.py:6-9`: "thermal relaxation for the gate duration …, then the gate-class
channel — **the same composition order as Qiskit-Aer's `thermal.compose(depolarizing)`
convention**".

Aer's actual convention is the opposite: `qiskit_aer/noise/device/models.py:230` reads
`return depol_error.compose(relax_error)`, i.e. depolarising first, relaxation second.
`tests/test_noise_model.py:134` builds the Aer side as `relax1.compose(depolarizing_error(...))`
— our order — so the cross-check confirms the implementation but not the attribution.

The numerical stake is small (the two orders differ by `p·(T(I/2) − I/2)`): on the 7-qubit
prototype network under ibm_fez, `TV = 6.6e-6` (`/tmp/chk_misc.py`).  Either order is defensible;
the pin is not.  **Fix**: say "we compose relaxation *before* the gate-error channel; Aer's
`basic_device_gate_errors` composes them the other way round (`models.py:230`); the difference is
`O(p·γ)`, measured at TV 6.6e-6 on the Fig-5 network".

### 13. `NoiseModel.scaled` silently changes the channel family, and `scaled(0)` raised at HEAD — minor

`src/ptqnn/noise/model.py:158-183`.  `scale_kraus` takes *any* Kraus set, reads off its average
fidelity, and returns a **depolarising** channel with the scaled fidelity:

```
before: amplitude_damping(0.1), 2 Kraus ops, F_avg = 0.96623
after  scaled(2.0): 4 Kraus ops, depolarising, F_avg = 0.93246
```

A ZNE-style sweep over a damping or coherent model therefore extrapolates a different channel
family from the one it started with, silently.  At HEAD `scaled(0.0)` also raised
`ZeroDivisionError` on `t1_us / factor` (fixed in the working tree by an early return).

**Fix**: document the replacement in the docstring (it is a real modelling choice: "we scale the
*average error*, re-expressed as depolarising"), and either refuse non-depolarising inputs or
provide `scaled_kraus_exact` for mixtures of unitaries where interpolation is well defined.

### 14. `_named` calls any 4-Kraus single-qubit channel "depolarizing" — minor

`src/ptqnn/noise/model.py:44-50`.  Measured: `pauli(0.1, 0.1, 0.1) -> "depolarizing"`, and
`thermal_relaxation(50, 90, 1.0, 0.02)` (Choi rank 4) -> `"depolarizing"`.  No correctness
impact — the orkan fast path recognises depolarising from the Choi matrix, not the name
(`backends/orkan.py:246-266`) — but noisy circuits printed in tests and transpiler output are
mislabelled, and `Channel.__hash__` mixes name into the hash.  **Fix**: reuse
`orkan.depolarizing_strength` (or a shared helper in `noise.channels`) instead of counting Kraus
operators, and fall back to `"channel"` when it says no.

### 15. `test_kwsa_map_fn_parallel_probes_equivalent` cannot fail — minor (test adequacy)

`tests/test_train.py:143-154`:

```python
par = kwsa.minimise(f, np.zeros(3), cfg, map_fn=lambda fn, xs: [fn(x) for x in xs])
```

That lambda is `kwsa._serial_map` written out; the test asserts a deterministic function equals
itself.  It says nothing about the contract that actually matters for a process pool — that
results come back **in probe order** and that `len(result) == len(xs)`.

**Fix**: use a `map_fn` that evaluates in a permuted order and restores it
(`idx = rng.permutation(len(xs)); vals = [f(xs[i]) for i in idx]; return [vals[list(idx).index(j)] …]`),
plus one that asserts the batch size, plus (marked `slow`) a real
`concurrent.futures.ProcessPoolExecutor` run.  While there: in `mode="coordinate"` `map_fn`
receives only **2** probes at a time (`kwsa.py:136`) and the third probe `f(x + step)` bypasses it
entirely (`kwsa.py:139`) — a pool will lose more to dispatch than it gains.  Either batch a whole
coordinate sweep's `±c` probes through `map_fn` (they are independent of each other, only the
step probe is not), or document that `map_fn` is only useful in `mode="full"`.

### 16. `TrajectoryBackend` results depend on `batch_size`, which is not in `name` — minor

`src/ptqnn/backends/trajectories.py:93-95, 105-115`.  Verified: same seed and
`n_trajectories = 1000`, `batch_size = 256` vs `1000` give **different** distributions, but both
report `name = "trajectories(n=1000, seed=0)"`.  Since `meta.json` records `backend` as that
string, two runs with the same recorded provenance are not reproducible from it.
**Fix**: include `batch_size` in `name`, or draw the branch randomness per trajectory rather than
per batch so the stream is batch-invariant (a `SeedSequence.spawn(n_trajectories)` would do it).

### 17. `git_hash()` is duplicated three times and never records a dirty tree — minor

`experiments/bitcount_figs.py`, `experiments/flappy_train.py:36-43`,
`experiments/iris_fig7_fig8_fig9.py:41-49` — three near-copies, two of which differ (the flappy
one passes `cwd=ROOT`, the Iris one inherits the caller's CWD and would silently record another
repo's HEAD).  None records `git status --porcelain`.  This review ran against a tree with three
modified source files; a run started now would stamp `git_hash: 871ee9e` while executing
different code.

**Fix**: one `experiments/_meta.py` with `run_meta(name, seed, backend, **extra)` returning
`{git_hash, git_dirty, python, timestamp, seed, backend, …}` and writing `meta.json`; all three
scripts call it.  That also fixes the fact that `bitcount_figs.py`'s meta records `seeds` while
the others record `seed`.

### 18. The Iris target ignores code multiplicity — minor (missing `[ASSUMPTION]`)

`src/ptqnn/tasks/iris/model.py:95-101`.  `target_distribution` puts one mass `(label+1)²` on each
*distinct* code, so a code carried by 8 samples has the same target probability as a singleton,
while `accuracy` (`model.py:346-359`) weights *per sample*.  The paper's Fig 8 says "each bin of
the histogram is a **dataset example**", which reads the other way.  The merge choice (mean
label) is flagged `[ASSUMPTION]`; the weight choice is not.
**Fix**: add the `[ASSUMPTION]` and offer `weights="uniform" | "multiplicity"` so the sensitivity
can be measured — with `merge_duplicate_codes` already returning `counts`, it is three lines.

### 19. Mutation audit: six mutations no test catches — major (test adequacy)

I mutated the sources one change at a time in a throwaway copy (`/tmp/ptqnn-mut`, its own
`uv sync` — a bare `cp -r` leaves the venv shebangs pointing at the original tree and every
mutation falsely appears uncaught) and re-ran `uv run pytest -m "not slow"` (baseline 163
passed).  37 mutations; **31 were caught**, including every one I most wanted caught: swapping
`p01`/`p10` and transposing `confusion_for`; disabling idle noise; `d/(d−1) → (d−1)/d`;
`(d·F_e+1)/(d+1) → (F_e+1)/2`; swapping `t1`/`t2`; always-branch-0 and skipping the branch
renormalisation in the trajectory sampler; the `marginal` and Iris bit orders;
`(label+1)² → (label+1)`; `merge_duplicate_codes` mean → max; the Flappy default angle map and
threshold direction; ignoring `restate_incumbent`; dropping the SU(2) normalisation in
`zyz_angles`; swapping control/target in `decompose_crx` and the `cz` expansion; and **all six**
bit-order mutations in the QASM/executor layer.  The hardware layer is the best-protected code
in scope.

The six that survive:

1. **`noise/model.py:143-149`** — emitting the gate channel *before* the relaxation channel (the
   composition order of finding 12) breaks nothing in a default run.  The only test that pins it
   is `test_matches_qiskit_aer_without_idling`, which is both `@pytest.mark.slow` **and**
   `importorskip("qiskit")` — so the ordering is unprotected in `scripts/check-all.sh` (which
   runs `-m "not slow"`) and on any machine without qiskit-aer.  Add a non-slow test that pins
   the *op order* of `nm.apply(circ)` for a model with both T1/T2 and a gate channel.
2. **`backends/gpu.py:120`** — cutting the GPU trajectory count 100-fold changes nothing:
   `test_gpu_trajectories_match_density_matrix` derives its 4σ band from the *nominal*
   `n = 20 000` and never checks the sampler used it, so the band is 10× too wide to notice.
   Only a 1000× cut fails.  Assert on `acc.sum()`/an exposed trajectory counter, or tighten the
   band by comparing two different `n` and checking the error falls as `1/√n`.
3. **`backends/gpu.py:87`** — `cp.outer(psi, psi.conj())` → `cp.outer(psi.conj(), psi)` (i.e. ρᵀ)
   is undetected because `test_gpu_unitary_and_initial_state` uses a real basis state
   (`v[5] = 1`), for which ρ = ρᵀ.  It is **not** an equivalent mutant: I checked that after one
   `crx` and an amplitude-damping channel the diagonals differ by 0.31 for a complex `psi`.  Use
   a complex, entangled `initial_state` in that test.
4. **`tasks/iris/model.py:230`** — flipping `kl_loss(restrict=False)` to `True` passes the
   non-slow suite, although `model.py:26-35` calls this choice load-bearing (charging the
   input-register leak vs post-selecting it away, and the difference between a 95 % classifier
   and a chance-level one).  `test_prototype_layer_leaks_the_input_register_and_support_post_selects_it`
   tests `model_distribution`'s `support=` argument, not `kl_loss`'s default.
5. **`tasks/flappy/train.py:128`** — deleting the shaping term (`return -ret` instead of
   `-ret + λ·err`) passes.  `test_objective_is_negative_shaped_return` builds its expectation by
   calling `tr.episode_loss` — the function under test — so both sides move together; and
   `test_objective_prefers_the_baseline_behaviour`'s `−1.0` margin is cleared unshaped.  Since
   the module docstring says shaping is what makes the loss differentiable at all
   (`train.py:8-14`), assert the two terms separately against hand-computed values.
6. **`backends/trajectories.py`** — `>=` → `>` in the cumsum comparison survives, but that is a
   near-equivalent mutant (it differs only on exact ties of a continuous uniform); low priority.

Weak or tuned tests found along the way, worth tightening while the above are fixed:

* `tests/test_gpu.py:121-122` — runs a 200-trajectory CPU backend and then asserts only
  `cpu.shape == est.shape`.  A shape check dressed as a cross-check.
* `tests/test_iris.py:151` — `0.5·start < median(kls) < 2.0·start`, a factor-4 band, for a claim
  the comment states as "the same decade".
* `tests/test_iris.py:190-192` — a 2-iteration smoke test asserting only
  `train_loss[-1] < train_loss[0]`; `:199-201` and `:230-231` assert shapes and ranges only.
* `tests/test_flappy.py:239` — `len(policy.decision_cache) <= 65*65`, where the bound *is* the
  grid size, so it is true by construction.  `:88`, `:95` — `15 ≤ steps ≤ 25` and
  `20 ≤ steps ≤ 35` for deterministic physics that has one right answer.
* `tests/test_presets.py:403` — `0 <= gate_2q_error < 0.2` spans three orders of magnitude;
  `:395-396` `pytest.skip`s when no device files are found, so
  `test_bundled_devices_load_and_are_physical` can pass vacuously if the package data breaks.
* `tests/test_noise_channels.py:89-90` — `len(bit_flip(0.05)) == 2` and
  `len(phase_flip(0.05)) == 2` check the Kraus *count* only, so swapping the two functions'
  bodies goes undetected (`test_per_gate_override_and_skip` only checks the label `"pauli"`,
  which both produce).

### 20. Smaller points — nit

* `experiments/bitcount_noise.py` (1257 lines) and `experiments/ecg_table1.py` are **untracked**
  while `experiments/runs/bitcount_noise/…` and `runs/ecg/…` are cited by reports — the run dirs
  outlive the script that made them.  Commit them.
* `scripts/check-all.sh` runs `mypy src` only; `tests/` and `experiments/` are `ruff`-checked but
  never type-checked, and `experiments/` is where the reproducibility plumbing lives.
* `reports/reproduction.md:77` — "8/8 optimiser seeds reach the same result".  Only two Flappy run
  dirs exist and neither records eight seeds; the only support is a docstring
  (`tasks/flappy/train.py:73`, "perfect on every optimiser seed we tried") with no count.
* `reports/reproduction.md:85-86` — ECG "(pending)" while `experiments/runs/ecg/20260820T162014/`
  carries a complete `results.json`.
* `hardware/executor.py:_QISKIT_2Q` lists `iswap`, but `transpile._expand_to_cx` has no expansion
  rule for it, so a circuit that survives `to_qiskit` can still fail `to_gateset`.  Make the two
  coverage sets one table.
* `QuantumFlappyPolicy` is a mutable dataclass whose `_columns` and `decision_cache` are computed
  in `__post_init__` from `params`; assigning `policy.params = …` afterwards leaves both stale
  with no warning.  Make `params` a read-only property or recompute on set.
* `agent.py:6-9` quotes Fig 10's printed angles (`RX(1.9537)`, `CRX(-1.0422, …)`) as exact.  They
  are read off the figure image and appear nowhere in `literature/QPrototypeLearning-5.txt`; per
  `CLAUDE.md` ("figures read from the paper carry the reading error") they should say so.
* `transpile.metrics` counts `Channel` ops in `depth`/`n_2q` because it never checks
  `isinstance(op, Gate)` — harmless today (transpile rejects channels) but a trap.
* `TrajectoryBackend(batch_size=0)` spins forever (`trajectories.py:105-114`: `b = min(0, …) = 0`,
  `done += 0`).  `n_trajectories` is validated in `__init__`, `batch_size` is not.  Also reported
  independently in `reviews/002b-codex-noise-adversarial.md` #6.

**Corroboration.**  `reviews/002b-codex-noise-adversarial.md`, written in parallel by Codex
against the same noise layer, independently reaches "NO BUG" on the four things this review also
verified by execution — Choi/Kraus vectorisation, thermal relaxation for `T2 > T1`, the
depolarising definition and fidelity conversion, and ASAP scheduling with 2q channel attachment.
Two of its findings overlap with this one (zero-error preset channels ≈ finding 5, generic
`scaled()` ≈ finding 13); its readout-validation and `batch_size=0` points are additional.

---

## Verdict

**Accept with required fixes; two blockers.**

The physics is in good shape and, where I could test it against an independent implementation, it
is *right to machine precision*: the ASAP idle schedule reproduces Qiskit's own scheduler
delay-for-delay, the full noisy density matrix matches Aer to 1.7e-15 on a circuit with genuine
idling, thermal relaxation matches Aer across the `T2 > T1` regime, the depolarising ↔ fidelity
conversion and the readout confusion orientation are both correct and correctly pinned to the IBM
TOML's own definitions, and the transpiler is phase-exact on every gate set including the
degenerate 1q cases I went looking for.  The trajectory sampler is correct at the edges the scope
asked about, the orkan depolarising recognition is done on the Choi matrix rather than by name,
and the QASM/counts bit-order story is right in all three directions.  The device TOMLs are the
best-documented artefact in the tree.

The two blockers are not about the physics.  `reports/hardware-plan.md` cites no run dir and, of
the numbers I could check, eight are wrong — including one that contradicts a number in the same
report and one that calls a populated run directory empty; a reader has no way to separate the
rows that reproduce exactly (§2.2, §2.3, §3.2 all do) from the rows that are guesses.  And the
GPU density-matrix backend is broken *right now* by the perf refactor `30564d9` — and
`scripts/check-all.sh` says so, loudly (`2 failed, 382 passed`), yet that commit and the three
after it went in anyway, with a commit message advertising "GPU 18 s → 1.5 s" for a code path
that raises `TypeError`.  The gate works; it was not heeded.

Of the majors, three are the same shape and worth treating together: information that exists in
the repo is silently thrown away — `initial_state` by the router (3), `readout.duration_ns` by
`metrics`/`_wall_clock_s` (4), "not published" by `parse_device` (5).  Each is a one-line default
that turns a known unknown into a confident answer.  Finding 4 is also the one place this review
changed its own mind: the *noise* model is right to leave readout duration out, because IBM's
asymmetric `p10` already is the decay (1.07× the T1 prediction) — but nothing says so, and the
*cost* model omits the same 1560 ns for no good reason at all.  Finding 7 is the bill for review 001 #5 coming due: three hand-rolled simulators
now exist because `Backend` cannot express what the tasks need, and none of them can carry noise,
which is the next phase.  Finding 8 is a coordination failure rather than a bug — the fix for
Flappy's rotating batch shipped, was never wired up, and the report still describes the
unfixed behaviour as a result.  The mutation audit (19) is reassuring where it matters most —
31 of 37 mutations die, including every bit-order and every fidelity-conversion mutation — but
the six survivors cluster in exactly the two places this review is least happy about: the GPU
backend and the shaping/`restrict` defaults that decide what the tasks actually optimise.

Suggested order: 2 (the GPU backend is red at HEAD) → 3, 4, 5 (silent defaults,
and 4 changes published cost numbers) → 1, 8, 9 (correctness of the record) → 19 (six unprotected
mutations, two of them in the GPU backend that 2 also touches) → 6, 10, 11 → 7 (the protocol,
before P9 lands more on top of it) → 15, 16 → the rest.

---

## Resolution (main session, 2026-08-20, session time-boxed by the user)

| # | Decision |
|---|---|
| B1 | **Open.** `reports/hardware-plan.md` numbers flagged wrong (TVD, P(out=1), "lowest 2q error", 2^{n+1}−2n−2, SWAP fractions, Garnet T2 sentence, $ figure) are NOT yet corrected; the report header now carries a warning. Next session: regenerate every number from `experiments/runs/bitcount_noise/20260820T145335Z/results.json` and `reports/hardware-compile-table.md`. |
| B2 | Fixed: `superoperator` seeds from the first Kraus term (GPU dispatch restored); GPU tests green when the GPU is not saturated by another job. |
| 3, 4, 6, 10, 11, 19 (hardware) | Fixed: initial_state carried through route()/compile_for (permuted by initial layout); readout_ns added to critical path (kept out of NoiseModel, reason documented); table generation moved to experiments/hardware_compile_table.py with meta.json; Braket cost lookup keyed on slug, block de-duplicated; gateset_for warns/raises on Willow/Ankaa (GENERIC fallback, caveat propagated); invariant tests n_2q = 2·n_CRX + 3·n_swap. 104 hardware tests. |
| 5 | Fixed: `parse_device` raises on unpublished gate/readout errors unless `allow_missing=True`; `DeviceSpec.missing_fields` records them. Experiment scripts that loop over `available_devices()` must pass `allow_missing=True` for `rigetti_ankaa_3` (or skip it). |
| 7 | Accepted as design debt: promote the tasks' batched statevector evaluators into `backends/` behind a batch method on `Backend` (also requested by the ECG/Iris/Flappy agents). Next session, first item. |
| 8 | `restate_incumbent` is implemented and tested but not wired into Flappy's seed-rotation path; docs still describe the pre-fix behaviour. Open. |
| 9 | Report wording to be corrected to the actual bin counts (14/20, 12/20, 14/18). Open. |
| 19 | Mutation survivors listed; open. |
