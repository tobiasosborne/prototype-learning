# Architecture

`ptqnn` is a small, literate stack: a circuit IR whose every op is a CP map, several backends that
turn a circuit into an outcome distribution, a noise layer that rewrites an ideal circuit into a
noisy one, one optimiser, four tasks, and a hardware layer that rewrites circuits for vendors. The
package is ~7.7 k lines; nothing in it is framework-dependent (numpy + scipy; scikit-learn only for
Iris k-means/TSNE; Qiskit, CuPy, wfdb optional).

## Package map

```
src/ptqnn/
├── circuits/
│   ├── ir.py          Gate (unitary + witness), Channel (Kraus), Circuit; rx/ry/rz/h/x/crx/cx/unitary
│   └── neurons.py     classic_neuron, prototype_neuron, Network (the paper's shallow net)
├── backends/
│   ├── base.py        Backend protocol: name, probabilities(circuit) -> p[2^n]
│   ├── reference.py   numpy statevector + density matrix — THE ORACLE
│   ├── batched.py     batched pure-state kernel (B, 2^n) behind every task fast path
│   ├── trajectories.py Monte-Carlo Kraus unravelling (noisy, > 12 qubits)
│   ├── gpu.py         CuPy versions of density matrix + trajectories
│   ├── orkan.py, _orkan_ffi.py   ctypes binding to ../orkan (C, OpenMP density matrix)
│   └── shots.py       ShotBackend: counts/N with common random numbers
├── noise/
│   ├── channels.py    Kraus zoo, Choi algebra, fidelities, ReadoutError
│   ├── model.py       NoiseModel (gate channels + ASAP idle relaxation), NoisyBackend
│   ├── presets.py     DeviceSpec from devices/*.toml -> NoiseModel
│   └── devices/       7 vendor TOMLs with per-field provenance + README
├── train/
│   ├── kwsa.py        Kiefer–Wolfowitz finite differences inside simulated annealing
│   ├── losses.py      KL, cosine, squared error
│   └── distribution.py joint/conditioned input–output observables [PTQNN Eq 4]
├── tasks/
│   ├── bitcount.py    Fig 3–6
│   ├── iris/          data (k-means code), model (dataset superposition), train (Fig 7–9)
│   ├── flappy/        env (headless clone), agent (policy, SPRT, render), train
│   └── ecg/           data (synthetic + MIT-BIH), model (encode, PSNR/SSIM), train (Table I)
└── hardware/
    ├── gatesets.py    vendor bases (IBM Heron, Quantinuum H, IonQ, IQM, Rigetti, generic)
    ├── transpile.py   CRX→CX expansion, basis rewrite, routing, metrics, compile_for(device)
    ├── qasm.py        OpenQASM 3 export + bit-order conversion
    └── executor.py    Executor protocol; Local, QiskitAer, IBMRuntime (dry run), Braket/Quantinuum stubs; cost model
```

## The three invariants

1. **Qubit 0 is the most significant bit and the first input.** `index = Σ_q b_q · 2^(n−1−q)`.
   Bit strings in `[PTQNN Fig 5]` (`01101`) read left-to-right as qubits 0..4. The paper's circuit
   figures are 1-based top-to-bottom: our qubit `q` is the paper's `q+1`. Everything that crosses
   a boundary with a little-endian world (orkan, Qiskit, OpenQASM) converts exactly once, at the
   boundary: `_orkan_ffi.msb0_permutation`, `executor.to_qiskit` (qubit reversal, so Aer keys
   need no re-ordering), `qasm.from_qasm_bitstring` (string reversal).
2. **Every op is a CP map.** `Gate` checks unitarity at construction (`_trusted` skips the check
   only for gates whose unitarity is a theorem — rotations, Paulis, `controlled(U)`); `Channel`
   checks `Σ K†K = I`. Backends that only do unitaries (`batched`, `run_statevector`) raise on
   a `Channel` — there is no silent slow path.
3. **The numpy reference is the oracle.** `orkan`, `trajectories`, `gpu`, `batched`, and the
   Qiskit-Aer executor are each pinned to `reference` on *full* distributions or Choi matrices,
   never on a marginal such as `P(out=1)`. The reference itself is pinned to a blind second
   implementation written by Codex (`tests/oracle/indep_sim.py`).

## The network (`circuits/neurons.py`)

`Network(n_inputs, n_hidden, hidden="prototype"|"classic")` has `n_inputs + n_hidden + 1` qubits:
inputs `0..n_in−1`, hidden `n_in..n_in+n_h−1`, output last. Each hidden neuron touches **all**
inputs; the output neuron is always a *classic* perceptron (`RX(bias)` then `CRX(w_j)` from each
hidden qubit) `[PTQNN Fig 4, 5, 10, 12]`.

* Prototype neuron `[PTQNN Fig 2]`: per input `i`, `RX(φ_i)` on the input, `CRX(θ)` input→out,
  `RX(−φ_i)` on the input (un-compute). One shared `θ` per neuron, no separate bias.
* Classic neuron `[PTQNN Fig 1]`: `RX(bias)` on out, `CRX(w_i)` from each input; inputs untouched.
* Both have `n_inputs + 1` parameters per hidden neuron, so prototype and classic networks of
  equal shape have equal `n_params` — the comparison the paper makes is fair by construction.

Flat parameter layout: hidden neuron by hidden neuron `(φ_0..φ_{n−1}, θ)` (or `(w…, bias)`),
then the output neuron `(w_0..w_{n_h−1}, bias)`. `Network.circuit(params, prep, initial_state)`
prepends the input-preparation ops (`H` for superposition, `RX(θ_i)` for real-valued features,
basis flips) or starts from an explicit `initial_state` (Iris dataset superposition).

## Data flow of one experiment

```
paper figure ─pin─▶ task module (target, prep, loss) ─▶ kwsa.minimise(loss)
                                   │                          │
                                   ▼                          ▼
                          Backend.probabilities      KWSAResult (best params, history)
                          (reference / batched /             │
                           NoisyBackend(dm) /                ▼
                           ShotBackend / gpu)     experiments/<name>.py writes
                                                  runs/<name>/<stamp>/{meta.json, results.json,
                                                  params.*, figs}  ─▶ reports/*.md, report_figures.py
```

Training code never imports a concrete backend; it takes a `Backend`. Swapping
`StatevectorBackend()` for `NoisyBackend(DensityMatrixBackend(), device("ibm_fez").noise_model())`
or `ShotBackend(inner, shots=1000)` is the entire mechanism behind the noise and shot studies.

## Fast paths

The tasks' inner loops (Flappy frames, ECG windows, Iris superposition) do not build a fresh
`Circuit` per sample. They call `backends.batched` (`basis_columns`, `product_input_states`,
`batch_probabilities`) to push a whole batch of pure states through one unitary circuit; Iris
additionally post-selects `out = 1` analytically (`iris.model.fast_joint`, which never simulates
the output qubit). Each fast path is tested against the plain `Network.circuit` + reference route
(`tests/test_batched_backend.py`, `tests/test_iris.py`, `tests/test_ecg.py`). Fast paths are
unitary-only; a noisy evaluation of the same task goes through `NoisyBackend` and the density
matrix or trajectories (`experiments/ecg_iris_noise.py`, `flappy_noise.py`).

## Where assumptions live

Anything the papers do not fix is marked `[ASSUMPTION]` at the point of use, with the reason.
The biggest ones: KW–SA schedule (`train/kwsa.py`), real-valued angle map `θ = 2·asin(√s)`
(pinned to `[SNN Fig 4]`, ablatable), Flappy clone physics and shaped return
(`tasks/flappy/{env,train}.py`), ECG source/noise/split (`tasks/ecg/{data,train}.py`), Iris split
and duplicate-code merging (`tasks/iris/data.py`), vendor error → depolarising mapping
(`noise/presets.py`), 40 ns 1q duration for IQM/Rigetti (`experiments/hardware_plan.py`).

## Non-goals

No autodiff, no parameter-shift SGD (planned in `PLAN.md`, never needed — KW–SA reproduces the
paper), no pulse-level noise, no live hardware submission (no credentials; executors dry-run).
