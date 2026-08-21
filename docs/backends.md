# Backends

A backend is anything with `name` and `probabilities(circuit) -> p` where `p` is the
length-`2^n` computational-basis distribution, qubit 0 = most significant bit
(`ptqnn/backends/base.py`). Training code depends on nothing else. All of them take the same
`Circuit` (including `initial_state`); the noisy ones additionally accept `Channel` ops.

## The matrix

| Backend (module) | State | Noise | Practical cap | Randomness | Pinned to |
|---|---|---|---|---|---|
| `StatevectorBackend` (`reference`) | pure, `(2,)*n` tensor | no — raises on `Channel` | ~25 q (memory) | none | blind oracle `tests/oracle/indep_sim.py`; closed forms |
| `DensityMatrixBackend` (`reference`) | `(2,)*2n` tensor, superoperator per op | yes | 12 q comfortable (4096²); 13–14 q slow | none | statevector (`|ψ⟩⟨ψ|`), oracle, Qiskit-Aer |
| `BatchedStatevectorBackend` (`batched`) | `(B, 2^n)` batch of pure states | no | as statevector × B | none | reference on full distributions, 770 adversarial cases (review 003) |
| `TrajectoryBackend` (`trajectories`) | batched pure states, one Kraus branch sampled per channel | yes (unbiased estimator) | 17–20 q at B=256 | `seed` | density matrix at 4 σ, ≥1000 trajectories |
| `GpuDensityMatrixBackend` / `GpuTrajectoryBackend` (`gpu`) | same numerics on CuPy | yes | 13–14 q DM in 24 GB (RTX 3090) | `seed` | CPU versions (`tests/test_gpu.py`) |
| `OrkanDensityMatrixBackend` (`orkan`) | orkan `MIXED_TILED` C buffer | 1q Kraus native; 2q via fast paths | ~14 q (2 GB) | none | reference to `_ATOL` on full DMs (`tests/test_orkan_backend.py`) |
| `ShotBackend` (`shots`) | wraps any of the above | inherits | inherits | `seed`; CRN | exact marginals at 1000+ shots; maximal coupling verified (review 003) |
| `NoisyBackend` (`noise/model.py`) | wraps a DM/trajectory backend | applies a `NoiseModel` + readout | inherits | inherits | Aer cross-check (`tests/test_noise_model.py`) |

Pick by task: exact reproduction → statevector (or `batched` inside task fast paths); noise at
≤ 12 q (bit-count, Flappy, ECG) → density matrix, GPU when sweeping; noise at Iris scale
(17 q+) → trajectories; finite shots → `ShotBackend` around whichever of those.

## Reference (`backends/reference.py`)

Written for obviousness: an `n`-qubit state is an array of shape `(2,)*n` whose axis `q` *is*
qubit `q`; a `k`-qubit operator is `tensordot`-ed into the matching axes and the axes moved
back. Density matrices are `(2,)*2n` (ket axes then bra axes); a gate acts as the superoperator
`U ⊗ U*` and a channel as `Σ K ⊗ K*` on the block, one matmul each. `run_statevector` refuses
channels. `sample_counts(p, shots, rng)` is the only place randomness enters a noiseless run.
Helpers: `probabilities`, `marginal(p, keep)`, `superoperator(kraus)`.

The reference is itself cross-checked against `tests/oracle/indep_sim.py`, a deliberately direct
simulator written blind by Codex from a brief that contained only the conventions
(`notes/briefs/codex-01-indep-oracle.md`), on random circuits with random 1q/2q unitaries and
channels (`tests/test_oracle_crosscheck.py`).

## Batched kernel (`backends/batched.py`)

Review 002 #7: Flappy, Iris and ECG had each grown a private `rx`/`crx` loop. They are one
computation — a batch `(B, 2^n)` of pure states through the same unitary circuit, one pass per
gate. `run_batch`, `batch_probabilities`, `basis_columns(circuit, qubits)` (the unitary's columns
on a sub-register), `product_input_states(amplitudes, n_qubits, n_inputs)` (a batch of product
inputs). Any 1q/2q gate the IR can express, any qubit order. Unitary-only by design.

## Trajectories (`backends/trajectories.py`)

Quantum-jump unravelling: at each channel, branch `i` with probability `‖K_i ψ‖²`, renormalise;
the mean of `|ψ|²` over `n_trajectories` estimates the noisy distribution with error
`√(p(1−p)/N)` per outcome. Unitary circuits are exact with one trajectory. Memory
`B · 2^n · 16` bytes. Used for Iris under noise (`experiments/ecg_iris_noise.py`, which sizes
`n_trajectories` from a wall-clock budget and a probe run).

## GPU (`backends/gpu.py`)

The reference is written against the NumPy API only and CuPy implements
`__array_function__`, so the same functions run on-device once state and gate matrices are
CuPy arrays; `gpu.py` only moves data. Two rules learned the hard way: keep `np.asarray` out of
dispatch paths (CuPy ≥ 13 raises; session 2 regression in `superoperator`), and on WSL2 set
`LD_LIBRARY_PATH=/usr/lib/wsl/lib` or CuPy reports `cudaErrorNoDevice` even though `nvidia-smi`
works. `gpu_available()` is safe at import time; tests skip without a device.

## orkan (`backends/orkan.py`, `_orkan_ffi.py`)

ctypes binding to the C simulator in `../orkan` (`OMP_NUM_THREADS` read at `dlopen`; the backend
sets 32 unless the variable is already set). `_orkan_ffi` mirrors the three C structs byte-exactly
and guards every argument in Python because orkan's validation macros call `exit()` — a missed
guard is process death without a traceback. Conventions: orkan's `rx/ry/rz` are `exp(−iθP/2)`
like ours (asserted in a test); orkan is lsb0, we are msb0, converted once at readout
(`msb0_permutation`). Every gate routed to a native kernel has its matrix compared to the IR's
first; a mismatch falls back to the generic `single_from_mat`/`two_from_mat`.

orkan has no 2-qubit channel. Three routes, most specific first: depolarising (recognised from
the Choi matrix; factorises into `k` single-qubit twirls — 32 ms vs 400 ms at 11 q), mixture of
unitaries (every Pauli channel; `len(kraus)` 2q gates), and general Kraus (pull the DM out, apply
with the reference, write back — slow, warned once, counted in `stats["slow_path_channels"]`).
`mixture_fast_path=False`/`native_gates=False` exist to test the generic paths.

Run `uv run pytest -m orkan`; tests skip when `liborkan.so` is absent
(`ORKAN_LIB` env var or `../orkan/cmake-build-release/src/liborkan.so`).

## Shots and common random numbers (`backends/shots.py`)

`ShotBackend(inner, shots, seed)` returns `counts/shots` with counts drawn from the inner
backend's exact distribution. On hardware this is all one ever gets; wrapping a backend makes
every training loop in the repo run on estimates unchanged.

`common_random_numbers()` is a context manager: inside it, the first evaluation is sampled
normally and every further evaluation is *maximally coupled* to it (`coupled_counts`), so two
probes whose distributions differ by total variation `TV` share all but a `TV` fraction of their
shots. `kwsa.minimise(probe_context=backend.common_random_numbers)` wraps exactly the `±c` and
proposed-step evaluations of each coordinate. Effect: the finite-difference slope variance drops
from `O(1/(c²N))` to `O(1/(cN))`; measured ÷2.9 at `c = 0.2`, ÷12–22 at `c = 0.05`. Shared
inverse-CDF uniforms (the obvious first attempt) gave only 1.3× — the coupling matters
(`tests/test_shots.py`, `reports/noise.md` "Shots").

## Agreement tests, and what they do not cover

Every backend-vs-reference test compares full distributions or full density matrices on random
circuits with random unitaries and channels; tolerances are stated in the test (machine precision
for exact backends, 4 σ with ≥ 1000 samples for stochastic ones). A sister project once shipped
a teleportation that moved only the diagonal and passed a `P(out=1)` test — hence the rule. Not
covered: performance regressions (timings live in docstrings and worklog, not tests).
