# Noise study — prototype vs classic QNNs under vendor-calibrated CP-map noise

Status: bit-count complete (11 qubits, exact density matrix on GPU); noise-aware training at
11 qubits and the Flappy/ECG/Iris noisy variants are in progress.  Full tables, figures and the
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

## Pending
* Study B at 11 qubits (now feasible: GPU evaluation 18 s → 1.5 s after the superoperator rewrite).
* Flappy / ECG / Iris under noise.
