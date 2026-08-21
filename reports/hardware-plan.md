# Hardware plan — running the prototype QNNs on real devices

Status: v2, 2026-08-21 (PLAN.md §1 item 4, phase P10; v1's numbers were found wrong in
review 002 blocker 1 and are regenerated here).  **Every number in this report is printed by
`experiments/hardware_plan.py` into `experiments/runs/hardware-plan/20260821T073211Z/numbers.md`** (meta.json: git hash, seed 17,
CuPy density-matrix backend, 118 s); the prose quotes that file and nothing else.
Calibration numbers come from `src/ptqnn/noise/devices/*.toml` (retrieved 2026-08-20,
per-field provenance with verbatim quotes; `devices/README.md` lists what is sourced and what is
missing).  Gate counts come from `transpile.compile_for` run against the real `DeviceSpec`s
(the gate-set × topology matrix is `reports/hardware-compile-table.md`).  Two inputs are
`[ASSUMPTION]`: IQM and Rigetti publish no 1q gate duration, so 40 ns is substituted
(`ASSUMED_1Q_NS`); Quantinuum publishes no T1/T2, so its idle error is charged from the
published *memory error per depth-1 circuit time*.  Nothing here has been submitted to a QPU;
no credentials exist in this repo.

---

## 1. Per-experiment resources

### 1.1 Logical circuits

Sizes from `docs/physics/rosenhahn_hirche_ptqnn.md`: bit-count 11 q (Fig 5) / 12 q (Fig 4);
Flappy 7 q (Fig 10); ECG 7/8/9 q (Fig 12, 1–3 prototypes); Iris 12 input qubits + hidden + output
⇒ 17–25 q *plus* dataset state preparation (§4.1).  One CRX per (input, hidden) pair plus one per
hidden→output edge; each CRX costs **2 native entanglers** (`transpile` docstring, exact identity).
Counts include the input preparation (RX per input for Flappy/ECG/Iris, H per input for the
bit-count superposition), which is why depth is one more than the bare network.

| experiment | qubits | logical depth | CRX | 1q |
|---|---:|---:|---:|---:|
| Flappy `Network(2,4)` | 7 | 14 | 12 | 19 |
| ECG 1 prototype `Network(5,1)` | 7 | 8 | 6 | 16 |
| ECG 2 prototypes `Network(5,2)` | 8 | 11 | 12 | 26 |
| ECG 3 prototypes `Network(5,3)` | 9 | 14 | 18 | 36 |
| bit-count `Network(5,5)` | 11 | 20 | 30 | 56 |
| bit-count `Network(5,6)` | 12 | 23 | 36 | 66 |
| Iris `Network(12,4)` | 17 | 24 | 52 | 109 |
| Iris `Network(12,12)` | 25 | 48 | 156 | 301 |

### 1.2 After transpilation (our transpiler, per real device preset)

`n_2q` = native entanglers including 3 per routed SWAP, with the SWAP share `3·n_swap/n_2q` in
brackets; `t_crit` = ASAP critical path from the preset's gate durations **plus readout** where
published (ibm_fez 1560 ns; nobody else publishes one).  Routing is **greedy nearest-neighbour**
(`transpile.route`) — see §3.2 for what qiskit's SABRE does instead.

| experiment | H1-1 / H2-1 (all-to-all) | IonQ Forte (all-to-all) | ibm_fez (heavy-hex) | IQM Garnet (square) | Ankaa-3 (square) | Willow (square) |
|---|---|---|---|---|---|---|
| Flappy 7 q | 24 · d36 · — | 24 · d62 · 17.9 ms | 90 (22 SWAP, 73 %) · d297 · 12.9 µs | 63 (13, 62 %) · d182 · 6.8 µs\* | 63 (13, 62 %) · d218 · 10.3 µs\* | 63 (13, 62 %) · d94 · 3.2 µs |
| ECG 7 q | 12 · d36 · — | 12 · d62 · 17.9 ms | 57 (15, 79 %) · d193 · 9.0 µs | 42 (10, 71 %) · d133 | 42 (10, 71 %) · d153 | 42 (10, 71 %) · d59 · 2.0 µs |
| ECG 9 q | 36 · d48 · — | 36 · d82 · 23.8 ms | 174 (46, 79 %) · d432 · 18.3 µs | 123 (29, 71 %) · d341 | 123 (29, 71 %) · d375 | 123 (29, 71 %) · d157 · 5.3 µs |
| bit-count 11 q | 60 · d60 · — | 60 · d102 · 29.7 ms | 327 (89, 82 %) · d676 · 27.8 µs | 207 (49, 71 %) · d485 · 18.1 µs\* | 207 (49, 71 %) · d533 · 25.4 µs\* | 207 (49, 71 %) · d219 · 7.4 µs |
| bit-count 12 q | 72 · d66 · — | 72 · d112 · 32.6 ms | 438 (122, 84 %) · d798 · 32.5 µs | 276 (68, 74 %) · d605 | 276 (68, 74 %) · d649 | 276 (68, 74 %) · d251 · 8.7 µs |
| Iris 17 q (network only) | 104 · d96 · — | 104 · d162 · 47.3 ms | 800 (232, 87 %) · d997 · 39.9 µs | 578 (158, 82 %) · d901 | 578 (158, 82 %) · d952 | 578 (158, 82 %) · d388 · 13.3 µs |
| Iris 25 q (network only) | 312 · d144 (H2 only) | 312 · d242 · 70.8 ms | 2949 (879, 89 %) · d3325 · 131 µs | > 20 q | 2004 (564, 84 %) · d2752 · 49 µs\* | 2004 (564, 84 %) · d1112 · 37.9 µs |

\* IQM publishes no 1q duration and Rigetti none for 1q; `t_crit` uses the `[ASSUMPTION]` 40 ns
PRX bound from IQM's Radiance sheet and 40 ns for Rigetti's ±π/2 RX.  Quantinuum publishes **no
gate durations**, so H1/H2 have no `t_crit` — a hole in any duration-based error budget.

### 1.3 Shot budgets — formulas

**(a) An activation `P(out = 1) = p`.**  `p̂ = k/N`, binomial, `σ = √(p(1−p)/N) ≤ 1/(2√N)`, so
for a 1-σ absolute precision `ε`:  **`N ≥ 1/(4ε²)`**.

| ε | 0.05 | 0.01 | 0.005 | 0.003 | 0.001 |
|---|---:|---:|---:|---:|---:|
| N | 100 | 2 500 | 10 000 | 27 778 | 250 000 |

**(b) The 32-bin bit-count histogram / KL.**  The model distribution is `P(x | out = 1)` over
`K = 2⁵ = 32` bins `[PTQNN Eq 4]`, so only `out = 1` shots count: `N_eff = N · P(out = 1)`, and
our trained model (`experiments/runs/bitcount_noise/20260820T145335Z` A/prototype, seed 1,
KL = 5.47e-4) has **`P(out = 1) = 0.4871`** (statevector, `numbers.md` §shots).  Target and estimate are both normalised, so the
first-order term of `KL(t ‖ m̂)` vanishes and to second order
`KL̂ ≈ ½ Σᵢ (m̂ᵢ − tᵢ)²/tᵢ ~ χ²_{K−1}/(2 N_eff)`, hence

> **shot-noise KL floor  `E[KL̂] = (K−1)/(2 N_eff) = 31 / (2·0.4871·N) = 31.8 / N`**,
> with s.d. `√(2(K−1))/(2N_eff) = 3.94/N_eff`.

| KL floor wanted | 5e-3 | 1e-3 | 5.5e-4 (our converged KL) |
|---|---:|---:|---:|
| shots N | 6 364 | 31 822 | **57 857** |

The cost tables below budget **64 000 shots** for bit-count — the floor rounded up to leave the
estimate one s.d. below the converged KL.
**(c) ECG.**  `PSNR = 10 log₁₀(1/MSE)` for a signal in [0,1] `[PTQNN Eq 6]`; shot noise alone
contributes `MSE_shot ≤ 1/(4N)`, so the **PSNR ceiling is `10 log₁₀(4N)`**: 2 500 shots per
window sample caps you at exactly 40.0 dB — the Table I number — and you need ~25 000 shots per
sample to push the shot-noise floor 10 dB below it.  At 1 800 samples that is **4.5 × 10⁷ shots**
per denoised trace.  ECG is therefore a simulator experiment, not a hardware one.

**(d) Flappy decision.**  The policy flaps iff `p > ½`; a frame with margin `δ = |p − ½|` flips
with probability `Φ(−2δ√N)`.  Over the 100 000 frames (439 distinct
grid states) the trained agent visits in 20 held-out episodes
(`experiments/runs/flappy_noise/20260820T154249Z/results.json` C/margins; parameters from
`experiments/runs/flappy/20260820T143753Z/params.npy`): **median δ = 0.0194**, p25 0.0102,
p10 0.0042, p1 0.0002.  Visit-weighted mean flip probability per frame:

| N | 1 000 | 4 000 | 10 000 |
|---|---:|---:|---:|
| P(flip) | 0.164 | 0.080 | 0.052 |

The boundary is *thin*: at 10⁴ shots/frame one decision in 19 flips from shot noise alone,
before any device error, and the 1 % tail (δ = 2e-4) is unresolvable at any realistic budget.  **"Perfect play" is not a reachable hardware deliverable.**  The
reachable one is the *activation map* — `P(out=1)` on a grid of observations, against simulator
and noise model: same circuit, a few dozen jobs, falsifiable.

---

## 2. Per-vendor path

### 2.1 Access, SDK, adapter, native set

| vendor / device | access route | SDK | our adapter | native set (`gatesets.py`) | what we emit |
|---|---|---|---|---|---|
| IBM `ibm_fez` (Heron r2) | IBM Quantum Platform / Qiskit Runtime; **Open plan free, 10 min QPU / 28 d** ([docs](https://quantum.cloud.ibm.com/docs/en/guides/plans-overview)) | `qiskit-ibm-runtime` (not installed) | `IBMRuntimeExecutor` — **dry run only**; builds the real ISA circuit | `rz, sx, x` / `cz` | ZXZXZ Euler: 5 1q, 2 SX per run; CX = (I⊗H)·CZ·(I⊗H) |
| Quantinuum H1-1 (20 q) / H2-1 (56 q) | Quantinuum Nexus (`qnexus`) or Azure Quantum; emulators H1-1E/H2-1E consume HQCs too | `pytket` + `pytket-quantinuum` (not installed) | `QuantinuumExecutor` — **stub**, lists the four missing pieces | `u1q, rz` / `zz, rzz` | `RZ(a+c)·U1q(b, π/2−c)` — 2 1q gates per run; 1 RZZ per CX |
| IonQ Forte (36 q) | Amazon Braket (`arn:…/qpu/ionq/Forte-1`), IonQ Quantum Cloud direct, or Azure Quantum | `amazon-braket-sdk` (not installed) | `BraketExecutor` — **stub** | `gpi, gpi2` / `zz` (Forte; MS is Aria) | 4 1q gates per run (`GPI2·GPI2·GPI·GPI`); 1 ZZ per CX |
| IQM Garnet (20 q) | Amazon Braket, or IQM Resonance direct (free tier 30 credits/mo) | `amazon-braket-sdk` / `iqm-client` | `BraketExecutor` — **stub** | `prx` / `cz` | 3 PRX per run (no native RZ!) |
| Rigetti Ankaa-3 (84 q) | Amazon Braket, or Rigetti QCS | `amazon-braket-sdk` | `BraketExecutor` — **stub** | `rx, rz` / `iswap` (verbatim), `cz` legacy | **we synthesise CZ, not iSWAP** — a verbatim Ankaa-3 job needs Rigetti's own rebase |
| Google Willow | **no access.** "Willow Early Access Program … currently not available to the public"; proposals closed 2026-05-15 ([TQI](https://thequantuminsider.com/2026/03/28/google-opens-early-access-to-willow-quantum-processor-invites-experimental-proposals/)) | — | — | `xy_microwave, virtual_z` / `cz` | preset exists for simulation only |

Common path for everything except IBM: `qasm.to_openqasm3` (Braket accepts OpenQASM 3; the
non-standard names carry exact `gate` definitions).  Bit-string order is `qubit 0 leftmost`,
matching `[PTQNN Fig 5]`; `counts_from_qasm` un-reverses vendor keys.

### 2.2 Expected circuit fidelity — the arithmetic

Model: `F ≈ Π_gates (1 − ε_g) · Π_qubits (1 − ε_idle) · Π_qubits (1 − ε_ro)` — the standard
multiplicative EPLG-style estimate.  Idle error per qubit is the average-gate-infidelity of
thermal relaxation over the critical path `t`: `λ = e^{−t/T1}`, `Λ = e^{−t/T2}`,
`F_avg = ½ + (2Λ + λ)/6`.  Where no T1/T2 is published (Quantinuum) we substitute the *Memory
Error per Depth-1 Circuit Time* `m` and charge `(1−m)^(n_qubits · depth)` `[ASSUMPTION]`.

**Flappy, 7 q** (numbers from the TOMLs; counts from §1.2; `numbers.md` §fidelity).  The
simulator column is the total-variation distance between the *compiled* circuit run on an exact
density matrix without and with `DeviceSpec.noise_model()`, averaged over five observations
(grid corners and centre); the Quantinuum presets carry no T1/T2, so their simulated TVD
contains no idle error at all — `F(idle)` charges it separately from the memory-error figure.

| device | F(2q) | F(1q) | F(idle) | F(readout) | **F** | simulator TVD |
|---|---:|---:|---:|---:|---:|---:|
| Quantinuum H2-1 | `0.9989^24 = 0.9739` | `0.999981^87 = 0.9983` | `(1−2.0e-4)^(7·36) = 0.9508` | `0.999^7 = 0.9930` | **0.918** | 0.022 |
| Quantinuum H1-1 | `0.99903^24 = 0.9770` | `0.999982^87 = 0.9984` | `(1−2.2e-4)^(7·36) = 0.9461` | `0.9977^7 = 0.9840` | **0.908** | 0.025 |
| IonQ Forte | `0.996^24 = 0.9083` | `0.9998^204 = 0.9600` | `t = 17.9 ms`, `T2 = 1 s` → 0.9590 | `0.995^7 = 0.9655` | **0.807** | 0.120 |
| Google Willow (sim only) | `0.9959^63 = 0.772` | `0.99947^107 = 0.9449` | `t = 3.18 µs` → 0.8723 | `0.992^7 = 0.9453` | **0.601** | 0.255 |
| ibm_fez (SABRE O2, §3.2) | `0.99610^37 = 0.8653` | `0.999771^100 = 0.9774` | `t = 3.17 µs` → 0.8971 | `0.99233^7 = 0.9474` | **0.719** | — |
| ibm_fez (our greedy router) | `0.99610^90 = 0.7033` | `0.999771^553 = 0.8808` | `t = 11.3 µs` → 0.6840 | 0.9474 | **0.401** | 0.387 |
| IQM Garnet | `0.9951^63 = 0.7338` | `0.9992^282 = 0.7980` | `t = 6.8 µs`, `T2ₑ = 21 µs` → 0.3824 | `0.97^7 = 0.8080` | **0.181** | 0.552 |
| Rigetti Ankaa-3 | `0.99^63 = 0.5309` | no 1q error published | no T1/T2 published | no readout published | **≤ 0.53** | 0.301† |

**bit-count, 11 q:**

| device | F(2q) | F(1q) | F(idle) | F(readout) | **F** | simulator TVD |
|---|---:|---:|---:|---:|---:|---:|
| Quantinuum H2-1 | `0.9989^60 = 0.9361` | 0.9958 | `(1−2.0e-4)^(11·60) = 0.8763` | `0.999^11 = 0.9891` | **0.808** | 0.032 |
| Quantinuum H1-1 | `0.99903^60 = 0.9434` | 0.9960 | 0.8648 | 0.9750 | **0.792** | 0.034 |
| IonQ Forte | `0.996^60 = 0.7862` | `0.9998^504 = 0.9059` | `t = 29.7 ms` → 0.8970 | 0.9464 | **0.605** | 0.158 |
| Google Willow (sim only) | `0.9959^207 = 0.4272` | 0.8666 | `t = 7.4 µs` → 0.6102 | 0.9154 | **0.207** | 0.344 |
| ibm_fez (SABRE O2) | `0.99610^97 = 0.6843` | `0.999771^275 = 0.9390` | `t = 6.07 µs` → 0.7228 | 0.9186 | **0.427** | — |
| ibm_fez (greedy) | `0.99610^327 = 0.2783` | 0.6329 | `t = 26.2 µs` → 0.2654 | 0.9186 | **0.043** | 0.434 |
| IQM Garnet | `0.9951^207 = 0.3618` | 0.4535 | `t = 18.1 µs ≈ 0.86·T2ₑ` → 0.030 | 0.7153 | **0.004** | 0.579 |
| Rigetti Ankaa-3 | `0.99^207 = 0.1249` | — | — | — | **≤ 0.12** | 0.345† |

`F(readout)` uses the mean of the two published flip probabilities, `(1 − (p01+p10)/2)^n`.
† Rigetti's simulated TVD is an *underestimate*: its preset carries only the iSWAP error — no
1q, no readout, no T1/T2.  For comparison, the *logical* (uncompiled) bit-count circuit under the
same presets moves by TVD 0.005–0.039 only (`reports/noise.md`; `numbers.md` §noise_run): the
2–14× larger TVD here is the price of the compiled circuit's 2 entanglers per CRX, the
resynthesised 1q gates and, on the lattices, the SWAPs.

### 2.3 Cost per shot and per job

Vendor pricing retrieved 2026-08-20.  A "job" below is one circuit at the stated shot count.

| device | pricing formula | source | Flappy 7 q, 4 000 shots | bit-count 11 q, 64 000 shots | $/shot |
|---|---|---|---:|---:|---:|
| IBM PAYG | **$96/min** ( = $1.60/s) of QPU runtime; Flex $72/min (≥ $30 000 prepay); Premium ≈ $48/min | [QCR 2026-03-20](https://quantumcomputingreport.com/ibm-offers-special-promotion-to-open-plan-users/) — **secondary source**, `ibm.com/quantum/pricing` 403s. `[ASSUMPTION]` | 52 ms QPU ⇒ **$0.08**, or **$48** with `cost_estimate`'s 30 s per-job overhead | 1.8 s QPU ⇒ **$2.85**, or **$51** with overhead | 2e-5 |
| IBM Open | free, 10 min / 28 d rolling | [IBM docs](https://quantum.cloud.ibm.com/docs/en/guides/plans-overview) | **$0** | **$0** | 0 |
| Quantinuum H1/H2 | `HQC = 5 + C·(N₁q + 10·N₂q + 5·N_m)/5000` (exactly the formula in `executor.cost_estimate`) | [Azure pricing](https://learn.microsoft.com/en-us/azure/quantum/pricing), [Nexus costing](https://docs.quantinuum.com/systems/trainings/helios/getting_started/costing.html) | 0.0794 HQC/shot ⇒ **322.6 HQC** | 0.186 HQC/shot ⇒ **11 920 HQC** | 0.079 / 0.186 HQC |
| — in USD | **no published $/HQC** (PAYG is "contact Sales@Quantinuum.com"); Azure Standard $125 000/mo → 10 000 HQC ⇒ **≈ $12.5/HQC** `[ASSUMPTION]` | Azure pricing page | ≈ **$4 030** | ≈ **$149 000** | ≈ $0.99 / $2.33 |
| IonQ Forte (Braket) | $0.30/task + **$0.08/shot** | [AWS Braket pricing](https://aws.amazon.com/braket/pricing/) | **$320.30** | **$5 120** | 0.08 |
| IonQ Forte (Azure AQT) | `$25.7899 + $0.0001645·N₁q·C + $0.001121·N₂q·C` (mitigation off); `cost_estimate` prints it when the device slug is not in the Braket table | [Azure pricing](https://learn.microsoft.com/en-us/azure/quantum/pricing) | $25.79 + 4 000·(204·1.645e-4 + 24·1.121e-3) = **$268** | 64 000·(504·1.645e-4 + 60·1.121e-3) + $25.79 = **$9 636** | 0.060 / 0.150 |
| IQM Garnet (Braket) | $0.30/task + **$0.00145/shot**; IQM Resonance direct from $0.30/s, free tier 30 credits/mo | [AWS](https://aws.amazon.com/braket/pricing/), [IQM](https://iqm.tech/products/iqm-resonance/) | **$6.10** | **$93.10** | 0.00145 |
| Rigetti | Ankaa-3 has **no current row** on the Braket pricing page; its successor Rigetti Cepheus-1-108Q is $0.30 + **$0.000425/shot**. Azure: $0.02 per 10 ms of job time | [AWS](https://aws.amazon.com/braket/pricing/), [AWS what's-new](https://aws.amazon.com/about-aws/whats-new/2026/04/amazon-braket-rigetti-cepheus) | ≈ $2.0 (Cepheus rate) | ≈ $27 | 0.000425 |
| Braket reservations | IonQ Forte $7 000/h, IQM Garnet $3 000/h, Rigetti Cepheus $4 100/h | [AWS](https://aws.amazon.com/braket/pricing/) | — | — | — |

**Queue / turnaround.**  Quantinuum runs a dynamic
[Fair Queue](https://docs.quantinuum.com/systems/user_guide/hardware_user_guide/fair_queue.html)
(priority from HQC accumulation and group weights, *no published wait time*) — plan hours-to-days
and iterate on the H1-1E/H2-1E emulators.  IBM publishes no wait times either; a `Session` holds
the QPU for a burst of jobs and is the only way to make a 100-job sweep tractable.  Braket QPUs
have *availability windows* (IonQ/IQM/Rigetti are not online 24/7) and no published queue
statistics; Braket Direct reservations buy a deterministic slot.  **No vendor publishes a
turnaround commitment.**

---

## 3. Recommended order

### 3.1 First: Flappy 7 q on an all-to-all trapped-ion machine

The compile table settles it.  The prototype network is complete-bipartite (every input touches
every hidden qubit), the worst possible fit for a lattice.  For Flappy: **24 entanglers and
0 SWAPs** on H1/H2/Forte versus **90 and 22 SWAPs** on heavy-hex, **63 + 13** on a square
lattice.  Routing overhead, not gate error, dominates — `n_2q = 2·n_CRX + 3·n_swap`, and on
heavy-hex 73 % of the entanglers are SWAP (62 % on the square lattices).  Staged:

1. **IBM Open plan, free, today.**  7 q fits; qiskit gets it to 37 CZ / depth 98 (§3.2) ⇒
   F ≈ 0.72.  Zero cost, so this is where the M3 + ZNE pipeline gets debugged.  Deliverable:
   `P(out=1)` on a 4×4 observation grid at 4 000 shots (16 jobs, ~1 s QPU total).
2. **IonQ Forte via Braket, ≈ $1 285** for the same grid at 1 000 shots (16 × ($0.30 + $80)).
   F ≈ 0.81, no SWAPs, no layout risk — the *confirmation* run.
3. **Quantinuum H2-1: best number, worst price.**  F ≈ 0.92 on the second-lowest 2q error of
   any preset (1.1e-3; H1-1's 9.7e-4 is lower, but its SPAM error is 2.3e-3 vs H2-1's 1.0e-3,
   which is why H2-1 wins the product), but ≈ 323 HQC for one 4 000-shot job — worth it for the headline figure
   only, and only with an allocation.  Dry-run on H2-1E first.

ECG 7 q (`Network(5,1)`, 12 entanglers) is *cheaper* than Flappy and rides along in the same
job — but §1.3(c) means hardware ECG is a single-window sanity check, not Table I.

### 3.2 IBM Heron with SABRE — measured, not guessed

Our router is greedy nearest-neighbour, no lookahead, no layout search (`transpile.route`).
qiskit's own transpiler, via `IBMRuntimeExecutor.dry_run_payload` (heavy-hex patch, basis
`rz,sx,x,cz`, seed 0):

| circuit | our greedy router | qiskit O1 | qiskit **O2** | qiskit O3 |
|---|---|---|---|---|
| Flappy 7 q | 90 CZ, depth 297 | 45 CZ, d117 | **37 CZ, d98** | 37 CZ, d98 |
| bit-count 11 q | 327 CZ, depth 676 | 129 CZ, d217 | **97 CZ, d184** | 107 CZ, d180 |

SABRE + VF2 layout + 1q resynthesis buys **2.4× fewer entanglers on Flappy, 3.4× on bit-count**
and 3–3.7× shallower circuits — in fidelity, 0.40 → **0.72** and 0.043 → **0.427** (F from the
ISA circuit's own gate counts and a critical path walked with ibm_fez's durations, `numbers.md`
§sabre), the
difference between "unusable" and "usable with mitigation".  **A real IBM submission must go
through `qiskit.transpile(..., backend=<live backend>, optimization_level=2)`, never our
router**, which stays the honest device-independent comparison instrument it was built to be
(`IBMRuntimeExecutor` already does the right thing).  O3 is *worse* than O2 on bit-count — check
both per circuit; O3's extra 1q merging does not pay when the 1q error is 17× below the 2q one.

So bit-count 11 q **is feasible on Heron** at F ≈ 0.43: enough for a qualitative histogram (the
32 bars span 6×, 0.0089…0.0536) but not for a KL comparison — device error alone puts the KL
floor at 10⁻²–10⁻¹, two orders above our simulated 5.5e-4.  Run it as a *shape* demonstration.

### 3.3 Iris is out of reach

The network alone is 17 q / 104 entanglers on H2 — fine.  The input is not: `[PTQNN §III.C]`
prepares *the whole dataset* as a superposition over 12 qubits, an arbitrary 2¹² = 4096-amplitude
state.  Dense amplitude encoding costs `2^{n+1} − 2n − 2 = 8 166` CNOTs (Shende–Bullock–Markov) or
`≈ 23/24·2^n = 3 925` (Plesch–Brukner); at H2-1's 1.1e-3 that is `0.9989^8166 ≈ 1e-4` resp.
`0.9989^3925 ≈ 0.013`.  Exploiting sparsity (the 120 training points carry amplitude) gives
`O(k·n)`; the leading term `k·n = 1 440` CNOTs alone is `0.9989^1440 ≈ 0.21`, before the
network's 104 and before the implementation constant.  **Best case ≈ 0.2 fidelity, on the best
machine available, with a state-preparation compiler we have not written** (`qasm.to_openqasm3` refuses circuits with an `initial_state`).  Iris stays a
simulator experiment; the paper itself concedes state preparation of a dataset "can be a huge
effort".

---

## 4. Error mitigation plan

1. **Readout calibration (M3-style).**  Presets carry *directional* readout error for IBM
   (`p01 = 3.91e-3`, `p10 = 1.15e-2`, a 3× asymmetry) and Quantinuum (`1.2e-3 / 3.4e-3`), so a
   symmetric bit-flip model is known-wrong for both.  Plan: tensored single-qubit confusion
   matrices from `2n` calibration circuits (14 Flappy, 22 bit-count) at 4 000 shots, then M3's
   iterative-solve unfolding restricted to the *observed* subspace — never the full `2^n × 2^n`
   inverse, which amplifies noise.  `ch.ReadoutError.apply` is the forward map to invert; the
   inverse belongs in a new `ptqnn/hardware/mitigation.py`.  Readout is 5–8 % of the total error
   budget on IBM/IonQ and ~25 % of the *residual* once the SABRE circuit is short.
2. **ZNE by gate folding.**  Fold each native entangler `G → G G† G` for noise scales λ = 3, 5
   (unitary folding preserves the ideal circuit exactly), fit `E(λ)` linearly/Richardson,
   evaluate at λ = 0.  Validate the extrapolant *offline* first: `NoiseModel.scaled(λ)` rescales
   depolarising `p`, `1/T1`, `1/T2` and readout together, so run simulated ZNE on the noisy
   backend, check the extrapolated `P(out=1)` recovers the ideal, then fold on hardware.
   Cost ×3.  Caveat: folding triples an already 30 ms IonQ circuit; on Quantinuum, where memory
   error scales with depth, folding scales the idle error too — correct for ZNE, but say so.
3. **Hardware fine-tuning with KW–SA.**  The paper's optimiser is gradient-free
   (`[PTQNN §II.C, SNN §III.A]`, `ptqnn/train/kwsa.py`): Kiefer–Wolfowitz central differences
   inside simulated annealing.  That is exactly what you want on a QPU — no parameter-shift, no
   ancillas, one circuit per evaluation.  The cost is the evaluation count: one KW sweep is
   `3·n_params` evaluations, and `Network(5,5)` has 36 parameters ⇒ 108 evaluations/sweep, 2 160
   for our 20-sweep reproduction.  At 64 000 shots that is 1.4×10⁸ shots — **$11 M on Forte**.
   So: *train in simulation, fine-tune on hardware.*  Freeze the prototype layers, re-tune only
   the 6 output-neuron parameters: 18 evals/sweep × 5 sweeps × 4 000 shots = 3.6×10⁵ shots ≈
   $29 000 on Forte, $520 on Garnet, ~free on IBM Open.  KW tolerates shot noise provided the
   difference step `c_n` stays above it: with `σ_p ≈ 1/(2√N)` the finite-difference SNR is
   `~ c_n·|∂p/∂θ|·2√N`, so `c_0 = 0.2` (`bitcount.SUPERPOSITION_CONFIG`) and `N = 4 000` give
   SNR ≈ 25 at a typical `|∂p/∂θ| ≈ 0.5`.  A *static* device bias cancels in the central
   difference and costs nothing; *drift between the two legs* does not — hence one job per pair
   (§6).

---

## 5. What the code still lacks for a live submission

| # | missing piece | where it goes | note |
|---|---|---|---|
| 1 | credentials plumbing (IBM token/CRN, Quantinuum login + API key, AWS keys + S3 bucket) | new `ptqnn/hardware/credentials.py`, env-var only, never in-repo | the three executors already enumerate exactly what is missing in their `NotImplementedError` text |
| 2 | live `qiskit-ibm-runtime` submission + `SamplerV2` result → our counts | `executor.IBMRuntimeExecutor.run` | payload builder is done; needs the service handle and a **live** backend so the ISA circuit sees the real coupling map, not `heavy_hex_patch` |
| 3 | **SABRE / vendor routing** as a first-class option | `transpile.route` (add `method="sabre"` delegating to qiskit) | §3.2: 2.4–3.4× fewer entanglers; our greedy router must not be used for submission |
| 4 | **iSWAP synthesis** for Rigetti verbatim mode | `transpile._cx_in_gateset` | `CX = (1q)·iSWAP·(1q)·iSWAP·(1q)`; today we emit CZ and `RIGETTI_ANKAA.notes` says so. Also: RX is restricted to ±π/2, ±π, so `_synth_1q` must not emit arbitrary RX |
| 5 | calibration pull (IBM `backend.properties()`, Braket `GetDevice.standardized`, IonQ `characterization_url`) | `ptqnn/noise/presets.py` + `devices/README.md` "How to refresh" | closes the T1/T2 holes for Rigetti and IQM and gives *per-qubit* data instead of one median |
| 6 | job polling / async handles (`JobStatus`, retry, partial results) | new `ptqnn/hardware/jobs.py`; `Executor` protocol needs `submit()`/`result()` beside `run()` | Braket tasks and Quantinuum jobs are asynchronous; a synchronous `run()` cannot express a 6-hour queue |
| 7 | result conversion for Braket/Quantinuum key orders | `qasm.counts_from_qasm` covers OpenQASM; pytket's `BackendResult` and Braket's `measurement_counts` need their own adapters | the reversal bug this guards against is silent |
| 8 | M3 readout unfolding + ZNE folding | new `ptqnn/hardware/mitigation.py` | §4; `NoiseModel.scaled` is the simulator-side twin, already present |
| 9 | sparse/dense state-preparation synthesis | `ptqnn/hardware/stateprep.py` | required for Iris; `to_openqasm3` currently raises on `initial_state` |
| 10 | a real `$/HQC` and IBM `$/min` | `executor._SOURCES` + `cost_estimate` | both are `[ASSUMPTION]` today (§2.3); the IBM $1.60/s already in the code equals the $96/min secondary figure, which is corroboration, not confirmation |

---

## 6. Risks, and what the simulator predicts

**Calibration drift between jobs.**  `ibm_fez.toml` records the same chip drifting from
CZ 2.85e-3 @ 68 ns (2024-07-04) to 3.90e-3 @ 84 ns (2025-02-26) to 2.61e-3 live — a 1.5× swing.
A KW central difference whose two legs land in different calibration windows measures drift, not
gradient.  Mitigation: submit `θ±c` in the *same* job; re-measure a fixed reference circuit at
the top and bottom of every sweep.

**Crosstalk is in none of our presets.**  `devices/README.md`: no device has a usable
`idle.zz_coupling_khz`.  IBM's 2q error is measured in *isolation* and therefore excludes
simultaneous-gate crosstalk — so §2.2 is optimistic for a wide, parallel circuit, and the
prototype network is exactly that (all hidden qubits driven at once).  IBM's own
EPLG₁₀₀ = 4.54e-3 for `ibm_fez` against a median isolated CZ of 2.61e-3 sizes the gap: **~1.7×**.

**Trapped-ion duration vs dephasing.**  Forte's 950 µs ZZ makes the 11 q bit-count a **29.7 ms**
circuit against T2 ≈ 1 s — 3 % of a coherence time, tolerable, but ZNE at λ=5 pushes it to
150 ms.  Quantinuum publishes *no* gate durations, only a Memory Error per Depth-1 Circuit Time
(2.0e-4, H2-1); over 11 qubits × 60 layers that is a **12 % idle error**, larger than the entire
2q budget (6 %).  On H-series, minimise *depth*, not entangler count.

**IBM heavy-hex routing overhead.**  §1.2: with our router 73 % (Flappy) / 82 % (bit-count) of
native entanglers are SWAP-borne; SABRE's 37 / 97 CZ against the 24 / 60 of an all-to-all
machine puts its share at 35 % / 38 %.  Structural — a complete-bipartite logical graph on a
degree-3 lattice — and the single largest lever on any superconducting device, worth 2.4–3.4×.

**Rigetti and IQM.**  Ankaa-3's preset has *only* a 2q error, and its 99.5 % headline is the
fSim gate, not the universal iSWAP (99.0 %) we would use — a 2× optimism trap.  Our routed
bit-count critical path on Garnet (18.1 µs, with the 40 ns 1q `[ASSUMPTION]`) is 86 % of its
T2-echo (21 µs) and twice its T2-Ramsey (8.5 µs), so the 11 q circuit is dead there
(F ≈ 0.004, simulated TVD 0.58); even Flappy's 6.8 µs is a third of T2-echo (F ≈ 0.18).  Neither is a
candidate.

**What the simulator predicts.**  `reports/noise.md` (run
`experiments/runs/bitcount_noise/20260820T145335Z`, copied into `numbers.md` §noise_run) answers
(a)–(c) for the *logical* 11 q bit-count circuit: under the Quantinuum H1-1 / H2-1 presets the
trained prototype model's output distribution moves by **TVD 0.0063 / 0.0049** from its clean
value (KL to target 8e-4 / 7e-4 against 5.5e-4 clean); ibm_fez 0.029, Garnet 0.039.  The
*compiled* circuits of §2.2 move 5–15× more (H1-1 0.034, H2-1 0.032, Forte 0.158, Fez greedy
0.434), because the compiled circuit has 2 entanglers per CRX plus the resynthesised 1q gates
and, on lattices, the SWAPs — so a hardware KL comparison is lost on every superconducting
device and survives only on H-series and, marginally, on Forte.  Expect the hardware bit-count
histogram to be visibly flattened toward uniform on every superconducting device and
recognisable on H1/H2 and Forte.

---

## 7. Provenance and assumptions (ground truth)

Every calibration number above traces to a `[[provenance]]` entry in the device TOML it came
from — source URL, source kind (vendor calibration page, vendor paper, third-party calibration
snapshot), retrieval date and a verbatim quote.  `experiments/hardware_plan.py` joins those
entries to every field it reads and **refuses to write a run** if any field lacks one
(`tests/test_hardware_plan_script.py::test_every_calibration_number_used_has_a_provenance_entry`);
the full per-field table (≈60 rows, with quotes) is `experiments/runs/hardware-plan/20260821T073211Z/numbers.md` §provenance.  What is
*not* vendor-published and enters as an `[ASSUMPTION]`:

| id | assumption | affects |
|---|---|---|
| A1 | IQM Garnet / Rigetti Ankaa-3 1q gate duration 40 ns (unpublished) | their critical path, F(idle), cost rows |
| A2 | Quantinuum idle error `(1 − m)^(n_qubits·depth)` from the published memory error per depth-1 time | F(idle) for H1-1/H2-1; their simulated TVD omits it |
| A3 | F(readout) = `(1 − (p01+p10)/2)^n` | readout column |
| A4 | IBM $1.60 per QPU second + 30 s per-job overhead | ibm_fez cost rows |
| A5 | Quantinuum ≈ $12.5/HQC (Azure Standard plan ÷ its HQC allotment) | USD column for H1/H2 |
| A6 | Braket per-shot prices and the Azure AQT IonQ formula, retrieved 2026-08-20 | Forte / Garnet cost rows |
| A7 | State-preparation CNOT counts: Shende–Bullock–Markov `2^(n+1)−2n−2`, Plesch–Brukner `≈23/24·2^n`, sparse leading term `k·n` (Gleinig–Hoefler) with k = 120 | §3.3 |
| A8 | Quantinuum HQC `N_m = 2·n_qubits` | HQC counts |
| A9 | Flappy simulated TVD averaged over 5 observations (grid corners + centre) | Flappy TVD column |

Known gaps in the vendor data itself (recorded in `devices/README.md`, not papered over): Rigetti
publishes no 1q error, no readout error, no T1/T2 (its fidelity rows are upper bounds);
Quantinuum publishes no gate durations and no T1/T2; IQM's T1/T2 are a third-party snapshot; IBM's
numbers are medians we computed from the per-qubit table of one dated calibration snapshot
(props_fez.json, 2025-02-26).
