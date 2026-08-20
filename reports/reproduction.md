# Reproduction report — Rosenhahn & Hirche, *Prototype Learning for Quantum Neural Networks*

Status: in progress (session 1, 2026-08-20). Every number below cites a run directory under
`experiments/runs/` (gitignored; `meta.json` carries seed, git hash, backend, wall time).
Paper pins `[PTQNN …]` refer to `docs/physics/rosenhahn_hirche_ptqnn.md`.

## Architecture as implemented

* Prototype neuron: per input `RX(φ_i) · CRX(θ) · RX(−φ_i)` `[PTQNN Fig 2]`; classic neuron:
  `RX(bias)` + `CRX(w_i)` `[PTQNN Fig 1]`; output layer always classic `[PTQNN Fig 4/5/10/12]`.
* Superposition training: `H` on inputs; model distribution `P(x | out = 1)`; loss `KL(target ‖ model)`
  `[PTQNN Eq 4, Fig 5]`.
* Optimiser: Kiefer–Wolfowitz central differences inside simulated annealing `[PTQNN §II.C, SNN §III.A]`,
  one *iteration* = one coordinate sweep = `3·n_params` circuit evaluations (the paper does not define its unit).
* Real-valued inputs: `θ = 2·asin(√s)` so that `P(1) = s` — pinned by `[SNN Fig 4]` (CRX(1.369) ↔ 0.4).

## Fig 3 — prototype activation surface

Run `experiments/runs/bitcount/20260820T135930Z/fig03_activation.png`. Prototype `(π, 0)`, `θ = π/2`,
inputs `RX(π q)`: corners (0,1) → 1.000, (1,0) → 0.000, (0,0) = (1,1) → 0.500, matching the caption
("maximized … if only one condition is met the activation is reduced … does not match, zero").
The paper's surface bottoms at ≈0.2 rather than 0 (its parameters are not given).  **Reproduced.**

## Fig 5 — bit-count histograms

Target read off the figure: `p(x) = (popcount(x)+1)/112`, confirmed on all 32 bars (±0.0002).
Trained model (seed 1): KL = 5.5e-4 (`fig05_histograms.png`); the paper's own model histogram has
residual KL ≈ 2.6e-3 against the same target.  **Reproduced.**

## Fig 6 — convergence

| | paper | ours (6 seeds) |
|---|---|---|
| superposition, KL at start | ≈0.055 | median 0.170 (random init, amplitude π) |
| superposition, first KL < 0.005 | ≈8 iterations | median sweep 9 (≈ 1000 evaluations) |
| superposition, final | ≈0 | 5/6 seeds < 1e-3 at 20 sweeps; seed 2 stalls at 1.4e-2 |
| iterative baseline, ≈0 by | ≈120 iterations | **not reached**: median 0.15 after 200 sweeps |

The iterative scheme (one sweep on the squared error of one random input per iteration) is our
construction `[ASSUMPTION]`; see `experiments/runs/bitcount/20260820T135930Z/results.md` and the
tuning note.  **Superposition: reproduced; iterative baseline: qualitative only (slower, as in
the paper) — discrepancy recorded.**

Tuning note (iterative baseline, 3 seeds × 300 sweeps, scratch sweeps 2026-08-20): step sizes
`a0 ∈ {2, 5, 20, 50, 100}`, `alpha ∈ {0, 0.3}`, coordinate and whole-vector modes, and a scaled
per-sample target (`target_scale ∈ {0.3, 0.5, 0.7, 1}`) all leave the median KL between 0.05 and
0.3; the best single run reached 0.046.  Per-sample squared-error updates on one input at a time
thrash between inputs; whatever the authors' iterative scheme was (SNN describes sample-wise
ℓ2 + KW–SA but gives no schedule), ours does not reach their ≈0-by-120 curve.

## Fig 7–9 — Iris
(pending)

## Fig 10–11 — Flappy Bird

Run `experiments/runs/flappy/20260820T142739Z/` (params.npy, best.gif, log.csv).  Architecture as
Fig 10: 2 inputs `RX(2·asin(√x))` (bird height, next-gap centre, normalised), 4 prototype
neurons, classic output; flap iff `P(out=1) > 0.5`.  Trained by KW–SA on the shaped episode
return over 5 training seeds (813 loss evaluations, 129 s): **110 pipes in 5000 frames (the cap)
on all 20 held-out seeds** — the clone's equivalent of the paper's "plays the game perfectly".
Our clone (`ptqnn/tasks/flappy/env.py`, all constants `[ASSUMPTION]`) uses flap −7 px/frame and
gap/spacing 120/180 px: the classic −9/100/144 is provably unsolvable for a memoryless policy
(flap overshoot 40.5 px > 38 px budget), which is not what the paper's demonstrator tested.
Negative result: rotating training seeds per iteration breaks KW–SA because the incumbent loss is
not re-scored (`kwsa.minimise` is built for stationary objectives).  **Reproduced (on our clone).**

## Fig 12–13, Table I — ECG
(pending)
