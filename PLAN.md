# Prototype Quantum Neural Networks — project plan

Status: DRAFT v1, 2026-08-20. Paper: Rosenhahn & Hirche, *Prototype Learning for
Quantum Neural Networks* (`literature/pdf/QPrototypeLearning-5.pdf`); prerequisite
paper [26] Rosenhahn, Osborne, Hirche, *Stochastic Neural Networks for Quantum
Devices*, arXiv:2602.22241 (`literature/pdf/2602.22241.pdf`).

## 0. What the paper actually specifies (extracted from text + figures)

Ground truth pins are `[paper §/Fig]` = QPrototypeLearning-5, `[SNN §/Fig]` = 2602.22241.

| Item | Specification | Pin |
|---|---|---|
| Classic stochastic perceptron | inputs q1,q2 → CRX(φ_i) onto q3; bias RX(θ) on q3 | Fig 1 |
| Prototype perceptron | per input i: RX(φ_i) on q_i, then CRX(θ) q_i→q_out, then RX(−φ_i) on q_i (un-compute). Same θ for all CRX of one neuron; bias folded into θ | Fig 2, §III.B |
| Network topology | input layer → N prototype neurons (each touches *all* inputs) → one output neuron which is a *classic* perceptron (bias RX + CRX from each hidden qubit) | Fig 4, Fig 5, Fig 10, Fig 12 |
| Bit-count task | 5 inputs, 6 hidden (Fig 4) / 5 hidden (Fig 5), 1 output; 12 (resp. 11) qubits | Fig 4, 5 |
| Superposition training | H on every input qubit; measure joint distribution; model distribution is P(inputs = x, out = 1) renormalised over x; loss = KL(target ‖ model) | Fig 5, Eq 4 |
| Bit-count target | bar heights read off Fig 5 (bottom): p(x) = (popcount(x)+1)/112 for 5 bits (0.0089, 0.0179, …, 0.0536) | Fig 5 |
| Convergence claim | superposition training: KL 0.055 → ≈0 in ~8 iterations; iterative sample-wise: ~120 iterations | Fig 6 |
| Optimiser | Kiefer–Wolfowitz central differences inside simulated annealing, Metropolis acceptance exp(−Δe/kT); no gradient descent | §II.C, Eq 3, SNN §III.A |
| Iris | k-means (k=3) per feature on train split → 12-bit one-hot-per-feature code, 12 input qubits; whole dataset prepared as superposition state (state preparation, not H); amplitude-encoded class label; ≈98 % accuracy after nearest-label rounding; train/test loss curves Fig 7 (classic overfits, prototype doesn't) | §III.C, Fig 7, 8, 9 |
| Flappy Bird | 2 input qubits via RX(angle) (bird height, gap position), 4 prototype hidden neurons (q3..q6), 1 output (q7) with bias RX + 4 CRX; decision flap/no-flap from P(out=1); trained model plays "perfectly" (score 450 000) | Fig 10, 11, §III.D |
| ECG denoising | sliding window of 5 noisy samples s(t−2..t+2) → 5 input qubits via RX(angle), 1–3 prototype neurons, 1 output; output probability = denoised s(t); metrics PSNR/SSIM; Table I: PTQNN 32.9/0.87, 40.00/0.93, 40.04/0.93 for 1,2,3 prototypes; CQNN 32.72/0.90, 38.20/0.90, 39.65/0.92 | Fig 12, 13, Table I, Eq 6–8 |

Under-specified (we must choose, document, and ablate):
- KW/SA hyper-parameters (a_n, c_n, T schedule, k) — not given in either paper.
- Real-valued → angle encoding for Flappy/ECG: Fig 12 shows RX(0.54, 0.78, 0.52, 0.70, 0.78) for signal values ≈0.17–0.25 → consistent with θ = π·s. Default: θ = π·s; ablate θ = 2·asin(√s).
- ECG source and noise: Fig 13 shows a smooth ~2-beat waveform in [0,1], 1800 samples, noise σ≈0.03–0.04. We will use a PhysioNet/MIT-BIH record (if offline-available) and a synthetic ECG fallback, additive Gaussian noise; report both.
- Flappy Bird: reward/episode protocol, how "training" was done (SA on episode score), observation discretisation, physics constants. We build our own deterministic headless clone (pygame-free) and document.
- Iris: state preparation of the dataset superposition; we use exact amplitude-encoding (Möttönen) in simulation and also an explicit statevector initialisation (equivalent in the noiseless case).
- "Iterative" baseline of Fig 6: sample-wise loss per basis state, one SA step per sample.

## 1. Deliverables

1. **`ptqnn`** — Python package (src layout, `uv`-managed, Python 3.12), typed, literate, TDD.
   - `ptqnn/circuits`: gate IR (RX, CRX, H, arbitrary 1q/2q unitaries, state-prep), prototype/classic neurons, network builder. Circuits are *channel lists* (every op is a CPTP map; unitaries carry an explicit witness — Sturm.jl's lesson).
   - `ptqnn/backends`: `numpy` reference (statevector + density-matrix, slow, literate — the ground truth), `orkan` (cffi binding to `../orkan` C library, density-matrix + Kraus, OpenMP on 64 cores), `trajectories` (Monte-Carlo Kraus unravelling for >12 qubits, e.g. Iris at 19–25 qubits), optional `cupy` GPU (RTX 3090) if time permits. All backends cross-checked on random circuits to 1e-10.
   - `ptqnn/noise`: CP maps as Kraus sets; channel algebra (compose, tensor, Choi, diamond-norm upper bound, CPTP verification); library: depolarising (1q/2q), amplitude + phase damping from (T1, T2, gate duration), thermal relaxation, coherent over-rotation, readout confusion, crosstalk ZZ idle; `NoiseModel` = rules mapping (gate, qubits) → channel + idle channels on spectators + measurement channel. Importers: QuantumHardware.jl TOML device files (`../QuantumHardware.jl/devices/*`), Qiskit-Aer `NoiseModel` JSON (optional dep), raw Quantinuum RB JSON. Vendor presets with per-field provenance (URL, retrieved date, sha) — IBM Heron r2/r3, Quantinuum H1/H2, IonQ Forte/Aria, IQM Garnet/Crystal, Rigetti Ankaa-3.
   - `ptqnn/train`: loss (KL, cosine, ℓ2, χ²), KW–SA optimiser (paper's), plus parameter-shift SGD baseline (SNN Eq 4), shot-noise-aware evaluation (finite shots), checkpointing, seeded determinism.
   - `ptqnn/tasks`: bitcount, iris, flappy (headless env + agent + optional render to PNG/GIF), ecg (data loader, windowing, PSNR/SSIM).
   - `ptqnn/hardware`: transpilation to vendor native gate sets (CRX → native decomposition), layout for linear/heavy-hex/all-to-all, shot budgeting, export to OpenQASM 3; execution adapters (qiskit-ibm-runtime, pytket-quantinuum, Braket for IonQ/IQM/Rigetti) behind a common `Executor` protocol, with a `dry_run` mode. No credentials assumed.
2. **Reproduction report** `reports/reproduction.md` — every paper figure/table regenerated: Fig 3, 5, 6, 7, 8, 9, 11 (frames), 13, Table I, with our numbers next to the paper's and discrepancies explained.
3. **Noise study** `reports/noise.md` — same experiments under each vendor preset and under swept depolarising/T1-T2 strengths; noise-aware training (train under noise vs. train clean/evaluate noisy); shot-noise scaling; which model (prototype vs classic) degrades more gracefully; circuit depth/2q-gate counts after transpilation.
4. **Hardware plan** `reports/hardware-plan.md` — concrete per-vendor path (access programme, SDK, native gates, qubit counts needed: Flappy 7q, ECG 7–9q, bit-count 11–12q, Iris 19–25q), cost/shot estimates, error-mitigation (readout M3-style inversion, ZNE for CRX depth), what to run first, and what the simulator predicts we will see.
5. Repo hygiene: `CLAUDE.md` (≤80 lines, haldane-conjecture style) mirrored as `AGENTS.md`; `STATE.md`; `WORKLOG.md` index + `worklog/` shards (Sturm.jl convention, 200–500 lines/shard, newest first) ; `HANDOFF.md`; `docs/physics/*.md` ground-truth distillations with equation/figure pins (PDFs local only, gitignored); `git init` with conventional commits.

## 2. Stack

- Python 3.12, `uv`, `numpy`, `scipy`, `cffi` (orkan), `pytest` + `hypothesis` (property tests: CPTP-ness, backend agreement, unitarity), `ruff`, `mypy --strict`, `pytest-cov`. Optional extras: `qiskit`, `qiskit-aer` (noise-model import + independent cross-check), `pytket`, `amazon-braket-sdk`, `cupy-cuda12x`, `matplotlib`, `neurokit2`/`wfdb` for ECG.
- Orkan (surveyed): C17, `liborkan.so` already built in `../orkan/cmake-build-release/src/`, OpenMP, statevector + packed/tiled density matrix, 1-qubit Kraus channels (`channel_1q`) on mixed states only, exported `single_from_mat`/`two_from_mat` for arbitrary 1q/2q unitaries (tiled DM only), no CRX, no sampling, no RNG, no Python binding (Sturm.jl binds it via `ccall` — `../Sturm.jl/src/orkan/ffi.jl` is the template for our ctypes binding). Density-matrix memory caps it at ~14 qubits (14q ≈ 2 GB). We use it as the fast noisy backend for bit-count/Flappy/ECG (≤12 q) and supply CRX (via `two_from_mat` or decomposition), 2-qubit channels (composed from native gates + 1q Kraus, or via our numpy path), and sampling in Python.
- Codex (`gpt-5.6-sol`, xhigh) used for: independent second implementation of the reference density-matrix simulator (blind cross-check), adversarial review of the noise-channel library, and derivations (parameter-shift for CRX, analytic bit-count solution). Subagents: sonnet/opus only.

## 3. Work breakdown (phases = worklog sessions; each ends with green tests + shard entry)

| # | Phase | Exit criterion |
|---|---|---|
| P0 | Scaffold: uv project, CLAUDE.md/AGENTS.md, STATE/WORKLOG/HANDOFF, docs/physics distillations of both papers, git init, CI script `scripts/check-all.sh` | `uv run pytest` green (smoke), lint/type clean |
| P1 | Reference backend (numpy): statevector + density matrix; gates; Kraus application; measurement/sampling; property tests | random-circuit agreement sv↔dm 1e-12; CPTP checks |
| P2 | Prototype & classic neurons, network builder, Fig 3 surface reproduced; analytic check of single neuron activation P(out=1) formula | Fig 3 regenerated; closed-form test |
| P3 | KW–SA optimiser + KL loss; bit-count superposition training; Fig 5/6 | KL < 1e-3 within ≤20 iters; iterative baseline slower (Fig 6 qualitative) |
| P4 | Orkan cffi backend + trajectories backend; benchmark; GPU optional | agreement with numpy; 12q density matrix step time measured |
| P5 | Iris (k-means binarisation, dataset superposition state, train/test curves, TSNE) Fig 7/8/9, ≈98 % | acc ≥ 95 % on held-out; overfitting contrast reproduced |
| P6 | Flappy Bird: headless env, observation→angles, SA over episode return, play loop, frame export | agent survives ≥ 10 000 pipes deterministically; GIF |
| P7 | ECG: data, windowing, 1/2/3 prototypes vs CQNN, PSNR/SSIM Table I, Fig 13 | within ~1 dB PSNR of Table I or discrepancy explained |
| P8 | Noise library: channels, NoiseModel, QuantumHardware.jl/Aer importers, vendor presets with provenance; Aer cross-check | channel tests; Aer vs ours agree on noisy circuits |
| P9 | Noisy experiments: all tasks × presets × strengths; noise-aware training; shots | `reports/noise.md` with figures |
| P10 | Hardware layer: transpile to native sets, OpenQASM3 export, executor adapters (dry-run), cost model; `reports/hardware-plan.md` | dry-run jobs validate against vendor schemas |
| P11 | Final reproduction report, cleanup, handoff | all reports, STATE.md final |

Parallelism: P4, P8 and P6 env can run as subagents/codex alongside P3/P5. Reviews: each phase gets one adversarial review agent (opus) before merge.

## 4. Risks / open questions for Tobias
- ECG data: which recording did the authors use? Unknown — we will use PhysioNet MIT-BIH record 100 (or synthetic) and state so; Table I will not match exactly.
- Flappy "6 days / 450 000" is a runtime claim we will not replicate; we target deterministic perfect play on our clone.
- Iris at 19–25 qubits is statevector-fine but density-matrix-infeasible → noisy Iris uses trajectories (exact in expectation) — acceptable?
- Hardware execution itself needs credentials/budget — plan only, unless you say otherwise.
