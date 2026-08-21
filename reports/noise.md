# Noise study — prototype vs classic QNNs under vendor-calibrated CP-map noise

Status (2026-08-21): bit-count complete (11 qubits, exact density matrix on GPU); Flappy complete
(7 qubits, exact density matrix); ECG/Iris run in a *reduced* configuration only (see below);
noise-aware training at 11 qubits **not run** (budget).  Full tables, figures and the
interpretation: `experiments/runs/bitcount_noise/20260820T145335Z/results.md`.  Device presets
and their provenance: `src/ptqnn/noise/devices/README.md`.  Noise model construction:
`ptqnn/noise/model.py` (depolarising with the vendor's average gate error, T1/T2 relaxation during
gates and ASAP-scheduled idling, readout confusion; cross-checked against Qiskit-Aer).

## Headline results (bit-count, Fig-5 network, 30 CRX, 11 qubits)

| finding | number |
|---|---|
| clean KL after training | prototype 5.5e-4, classic 1.15e-3 |
| noisy KL, best → worst preset | Quantinuum H2-1 0.0007 → IQM Garnet 0.0062 (prototype); 0.0014 → 0.0084 (classic) |
| displacement KL(clean‖noisy), median over presets | 0.0021 (prototype) vs 0.0022 (classic) |
| noise-aware training (7-qubit network), recovery factor | median ×158 (prototype), ×361 (classic) |
| KL vs noise scale s (ibm_fez ×s) | KL(clean‖noisy) ∝ s^1.8, TV ∝ s^0.95 — linear response |
| shot-noise bias of plug-in KL | +0.034 at 1000 shots, +0.0035 at 10 000 (only ≈48 % of shots survive `out=1`) |
| 2q-gate budget after transpilation | 60 (all-to-all) / 327 (heavy-hex) / 540 (linear) native entanglers for 30 CRX |

Interpretation: the prototype network is better under noise on every preset, but its advantage is
inherited from the better noiseless fit — both layers have identical two-qubit gate counts and the
noise displaces both distributions equally.  Noise-aware training folds the channel into the
angles (the recovered parameters are *worse* on a noiseless simulator).  Two-qubit error dominates
the budget on 6/7 presets before routing and on all presets after routing to a lattice.

## Flappy Bird under noise (7 qubits, exact density matrix)

Run `experiments/runs/flappy_noise/20260820T154249Z/results.md`.

| finding | number |
|---|---|
| perfect games under every vendor preset (20 held-out seeds) | 100 % on all 7 devices; 0.2–1.0 % of grid decisions flip |
| ibm_fez scaled ×2 / ×4 | 60 % / 0 % perfect games (median score 110 / 3) |
| shot noise, clean policy | 1000 shots: 0 % perfect (median 28 pipes); 4000: 90 %; 10 000: 100 % |
| shot noise + ibm_fez | 4000 shots: 55 % perfect; 10 000: 70 % |
| decision margin δ = \|P(out=1)−½\| | median 0.019, p10 0.004 — flips/frame ≈ Φ(−2δ√N) |
| noise-aware fine-tuning (5 sweeps, ibm_fez) | no change needed: already 100 % |

Interpretation: gate noise at today's calibrated levels does not break the policy (the trained
decisions sit ≥ 0.01 from threshold for most visited states); **shot noise does** — the policy is
a thresholded expectation value with a ~0.02 margin, so ≥ 10⁴ shots per frame are needed for
perfect play, which is the binding constraint for any hardware demonstration (see
`reports/hardware-plan.md`).

## ECG and Iris under noise (reduced run — indicative only)

Run `experiments/runs/ecg_iris_noise/20260820T155618Z/results.md` (its STATUS block lists exactly
what was cut).  ECG: 9-qubit exact density matrix, **8 test windows** (samples 1347–1354, the
second beat of Fig 13) instead of the 898-window test block, 2 presets (`ibm_fez`,
`quantinuum_h2_1`) instead of 7; Iris: 17-qubit trajectories, **16 trajectories** per evaluation
instead of 2000 (per-outcome standard error ≈ 12 %), 2 presets, perfect dataset state preparation
assumed.  Nothing below is publication-grade; the numbers are quoted so the reduction is on
record, not as findings.

| finding (reduced block) | number |
|---|---|
| ECG PSNR loss under `quantinuum_h2_1`, 6 cells | −0.1 … +0.4 dB (within the 8-window noise) |
| ECG PSNR change under `ibm_fez` | CQNN −1.6 … −2.7 dB; PTQNN −0.9 … **+3.8 dB** — a gain under noise on 8 windows is a small-sample artefact, not a result |
| ECG shot budget for ≤ 1 dB loss (closed form `MSE_∞ + E[p(1−p)]/N`) | 10 000–30 000 shots per window; at 500 shots every cell collapses to ≈30.5 dB (the shot floor) |
| Iris prototype test accuracy, 16 trajectories | 0.95 → 0.64 ± 0.44 (`ibm_fez`), 0.95 (`quantinuum_h2_1`); classic 0.95 on both |
| ECG noise-aware fine-tuning, 1 sweep, 1 prototype | +2.7 dB under `ibm_fez`, **+3.7 dB noiseless** — the sweep finishes the clean optimisation, it does not fold the channel in |

Measured cost for the full grid (why it was not run): one 9-qubit noisy window ≈ 0.1 s → the
898-window block × 7 presets × 3 extra scales ≈ 4–6 CPU-hours; one 17-qubit noisy trajectory
0.3–0.7 s → a 2000-trajectory Iris evaluation 10–25 min, 34 evaluations ≈ 8 h.  Both exceed the
1–2 h session ceiling and stay unrun; the script is complete and tested
(`tests/test_ecg_iris_noise_script.py`).

## Not run / pending
* Noise-aware training at 11 qubits: measured at ≈1.3 h per (device, model) cell on orkan
  (8 threads, 2.9 s/eval); ≈55 min per cell on the GPU density matrix (1.5 s/eval × 108 evals ×
  20 sweeps); the 14-cell grid is ≈13 h.  **Not run** — exceeds the session ceiling; the 7-qubit
  noise-aware study (B, above) is the evidence on record.  Script supports it
  (`--studies B --b-network full`, progress.log, partial results).
* ECG/Iris full grid: see the reduced-run section above.
