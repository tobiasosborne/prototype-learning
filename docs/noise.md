# Noise layer

Three modules: `noise/channels.py` (CP maps as Kraus lists, Choi algebra, the standard zoo),
`noise/model.py` (`NoiseModel`: where channels go in a circuit; `NoisyBackend`), and
`noise/presets.py` + `noise/devices/*.toml` (vendor calibration → `NoiseModel`, with provenance).
Results of applying it: `reports/noise.md`; hardware consequences: `reports/hardware-plan.md`.

## Channels (`channels.py`)

Kraus sets are plain lists of `2^k × 2^k` arrays, composable before a qubit assignment is known;
`ir.Channel(qubits, kraus)` wraps one. Algebra: `apply`, `compose(second, first)`, `tensor`,
`choi`, `choi_to_kraus` (minimal set by eigendecomposition), `is_trace_preserving`. Channels are
compared through Choi matrices (`C = Σ_ij |i⟩⟨j| ⊗ E(|i⟩⟨j|)`, `Tr C = d`), never through one
output state.

Fidelities: entanglement fidelity `F_e = Σ |Tr K_i|² / d²`, average gate fidelity
`F_avg = (d F_e + 1)/(d + 1)` (Nielsen 2002). `depolarizing_probability(F_avg, n)` inverts
`F_avg = 1 − p (d−1)/d` — the conversion every vendor error number goes through.

Zoo: `depolarizing(p, n)`, `depolarizing_from_fidelity`, `amplitude_damping(γ)`,
`phase_damping(λ)`, `pauli`, `bit_flip`, `phase_flip`, `coherent_overrotation(axis, ε)` (a single
unitary Kraus operator), `thermal_relaxation(t1, t2, duration, excited_population)` (requires
`T2 ≤ 2 T1`). `ReadoutError(p01, p10)` (or `per_qubit`) is a classical confusion map applied to a
probability vector after simulation; validated at construction (Codex review 02b).

## NoiseModel (`model.py`)

`NoiseModel(one_qubit, two_qubit, per_gate, t1_us, t2_us, duration_1q_ns, duration_2q_ns,
durations_ns, readout, idle_noise, name)`. `apply(circuit)` returns a new circuit with `Channel`
ops inserted:

* **after every gate**, on its qubits: thermal relaxation for the gate duration (if T1/T2 set),
  then the gate-class channel — Aer's `thermal.compose(depolarizing)` order, so the two agree;
* **before a gate**, on each of its qubits that has been waiting: an idle relaxation channel for
  the wait. `schedule(circuit)` is ASAP with one clock per qubit (what a list scheduler and
  barrier-free hardware do);
* **after the last gate** all qubits idle until the latest clock (simultaneous measurement).

Readout is applied by `NoisyBackend(inner, model)` to the inner backend's distribution.
`scaled(f)` multiplies every strength (`p`, `1/T1`, `1/T2`, readout) by `f` for sweeps; gate
channels are re-emitted as depolarising of the scaled infidelity, so a non-depolarising channel is
*approximated* when `f ≠ 1` (and `f = 1` returns the model untouched — review 02b #8).

`[ASSUMPTION]` stated in the module: vendor RB numbers are average fidelities that already contain
relaxation during the gate; modelling the incoherent part as depolarising *and* adding T1/T2 over
the gate duration double counts slightly, on the pessimistic side. Same practice as
`Aer NoiseModel.from_backend`.

Cross-check: `tests/test_noise_model.py` builds the same model in Qiskit-Aer and compares
noisy distributions (needs `--extra qiskit`; skipped otherwise). A Codex adversarial review of
this module found three defects (readout validation, a `batch_size=0` infinite loop, `scaled(1)`)
and confirmed the Choi/Kraus, T2>T1-vs-Aer, depolarising/fidelity and ASAP semantics
(`reviews/002b-codex-noise-adversarial.md`).

## Vendor presets (`presets.py`, `devices/`)

`device("ibm_fez")` loads `devices/ibm_fez.toml` into a `DeviceSpec` (qubits, topology, native
gates, T1/T2, 1q/2q error + duration, readout p01/p10, notes, provenance, `missing_fields`) and
`.noise_model()` maps it: gate error `ε` → depolarising with `F_avg = 1 − ε` (1q and 2q
separately); median T1/T2 + durations → relaxation during gates and idling; readout `p01`/`p10`
when published, else the symmetric median for both. A device with unpublished 1q/readout error
(Rigetti Ankaa-3) raises unless `allow_missing=True`, so an optimistic model is never built by
accident. `available_devices()` lists the seven slugs.

| slug | what it is | notable gaps |
|---|---|---|
| `ibm_fez` | Heron r2, 156 q heavy-hex; medians we computed from IBM's per-qubit calibration JSON (2025-02-26) | only device with per-qubit arrays |
| `quantinuum_h1_1`, `quantinuum_h2_1` | 20 q / 56 q trapped-ion, all-to-all; live Performance Validation page + data sheets | no T1/T2 (memory error per depth-1 time instead), no gate durations |
| `ionq_forte` | 36 q all-to-all; vendor spec panel + arXiv durations | T1/T2 only as order-of-magnitude ranges |
| `iqm_garnet` | 20 q square; Braket page + IQM paper | T1/T2 third-party; 1q duration only a bound (40 ns assumed in `hardware_plan.py`) |
| `rigetti_ankaa_3` | 84 q square; press release only | no 1q error, no readout, no T1/T2 — `allow_missing` required |
| `google_willow` | 105 q square; Nature 2025 SI | not accessible; included for comparison |

Provenance rules (`devices/README.md`, authoritative): no number without a verbatim quote, URL
and retrieval date; fidelity→error conversions stated in the field line; ranges and bounds stay in
`notes`, never become values; one calibration epoch per file; `*_median` that is really a mean
says so. `experiments/hardware_plan.py::section_provenance` joins every number the hardware
report uses back to a provenance entry by exact field id *and value* and refuses to run on an
unsourced field (review 003 #2 — the strict version immediately found two parser gaps the
substring version had hidden).

What no preset captures: per-qubit spread (sweep, don't trust a median), crosstalk/ZZ, coherent
errors, leakage, non-Markovianity. `devices/README.md` records where each would have to come
from and how to refresh each vendor.

## How the noise studies use it

* Train clean, deploy noisy: `NoisyBackend(DensityMatrixBackend() | GpuDensityMatrixBackend(), dev.noise_model())`
  on the trained parameters of the reproduction runs (`experiments/bitcount_noise.py` study A,
  `flappy_noise.py`, `ecg_iris_noise.py`).
* Noise-aware training: the same `NoisyBackend` handed to KW–SA (`bitcount_noise.py` study B;
  run at 7 q only — 11 q is ≈ 55 min/cell on GPU and was not run, `STATE.md`).
* Strength sweeps: `model.scaled(s)` for `s` in a grid (study C; response is linear,
  `KL(clean‖noisy) ∝ s^1.8`, `TV ∝ s^0.95`).
* Shots: `ShotBackend` (study D; the plug-in KL estimator is biased by `(K−1)/(2·P(out=1)·N)`,
  which is what limits sampled training — `docs/training.md`).
* Transpiled circuits: `hardware_plan.py` simulates the *compiled* circuits (5–15× the logical
  TVD) — logical-circuit noise numbers understate hardware.
