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

Runs `experiments/runs/iris/20260820T144849Z` (seed 0), `…T145454Z`, `…T145455Z` (seeds 1, 2); 40
sweeps each, 12 input + 4 hidden + 1 output = 17 qubits, dataset loaded as a uniform superposition
over the distinct 12-bit codes (`Circuit.initial_state`).  Ground-truth finding: Fig 8's bar heights
are 1/86, 4/86, 9/86 (train) and 1/154, 4/154, 9/154 (test), i.e. **p(x) ∝ (label+1)²** — the
amplitude carries the label, the histogram its square (pinned in `tasks/iris/model.py`).

| | paper | ours (seeds 0/1/2) |
|---|---|---|
| test accuracy, prototype | ≈98 % | 95 / 91 / 98 % (encoding ceilings 97 / 99 / 99 % on the three splits) |
| test accuracy, classic | — | 95 / 92 / 91 % |
| Fig 7 classic | train → 0, test rises to ≈0.045 | train → 0 within 1–4 sweeps; test KL ends at 0.033 / 0.028 / 0.043 after bottoming earlier (seed 1: 0.024 at sweep 4, seed 2: 0.035 at sweep 2) |
| Fig 7 prototype | both descend to ≈0.01/0.02 | train 0.003–0.014, test monotone, minimum at the last sweep |
| Fig 9 TSNE | prototype weights span the data | **not reproduced** — both weight sets cluster away from the data |

Assumptions: 1/3 stratified train split — **chosen, not derived from Fig 8**: the paper states no
split; Fig 8 shows 15 (train) / 30 (test) non-empty bins, our 50/100 splits give 14/20, 12/20 and
14/18 distinct codes (seeds 0/1/2; `meta.json` `split.distinct_*_codes`), so the train side is a
near-match and the test side is not (a 12-bit one-hot-per-feature code simply does not spread 100
samples over 30 codes on any split we tried; review 002 #9).  Collisions merged by mean
label, KL renormalised over all 4096 inputs (post-selecting on dataset codes lets the prototype net
exploit inter-sample interference: train KL 1e-5 but read-out at chance).  **Fig 7/8: reproduced
qualitatively; accuracy in range; Fig 9: discrepancy.**

## Fig 10–11 — Flappy Bird

Run `experiments/runs/flappy/20260820T143753Z/` (earlier run 20260820T142739Z identical outcome) (params.npy, best.gif, log.csv).  Architecture as
Fig 10: 2 inputs `RX(2·asin(√x))` (bird height, next-gap centre, normalised), 4 prototype
neurons, classic output; flap iff `P(out=1) > 0.5`.  Trained by KW–SA on the shaped episode
return over 5 training seeds (1576 loss evaluations, 173 s; 8/8 optimiser seeds reach the same result): **110 pipes in 5000 frames (the cap)
on all 20 held-out seeds** — the clone's equivalent of the paper's "plays the game perfectly".
Our clone (`ptqnn/tasks/flappy/env.py`, all constants `[ASSUMPTION]`) uses flap −7 px/frame and
gap/spacing 120/180 px: the classic −9/100/144 is provably unsolvable for a memoryless policy
(flap overshoot 40.5 px > 38 px budget), which is not what the paper's demonstrator tested.
Rotating training seeds: in session 1 a rotating 5-of-10 seed batch gave 0/25 perfect games
because `kwsa.minimise` carried a stale incumbent loss across batches; with the incumbent
re-scored every sweep (`restate_incumbent`, now switched on automatically by `train(batch > 0)`)
and the annealer's final point reported, the same protocol gives **25/25 perfect games**
(`experiments/runs/flappy/batch5`, 86 s, 1 601 evaluations) against 25/25 for fixed seeds on the
same 25 held-out episodes (`experiments/runs/flappy/batch0`, 113 s, 1 576) — the earlier
negative result was an optimiser artefact, not a property of the task (review 002 #8).
**Reproduced (on our clone).**

## Fig 12–13, Table I — ECG

Runs `experiments/runs/ecg/20260820T162014` (synthetic, Table I grid), `…T165910` (encoding
ablation), `…T165912` (PhysioNet MIT-BIH 100).  Architecture as Fig 12: 5-sample window →
`RX(2·asin(√s))`, 1–3 hidden neurons, classic output, denoised value = `P(out=1)`.  Signal:
Fig-13-tuned Gaussian PQRST waveform (McSharry form), σ = 0.02 additive noise (noisy input
32.0 dB), contiguous 50/50 split with guard band; PSNR with R = max−min of the clean signal.

| neurons | CQNN PSNR paper / ours | PTQNN PSNR paper / ours | SSIM paper / ours (windowed) |
|---|---|---|---|
| 1 | 32.72 / 36.13 | 32.9 / **37.41** | 0.90, 0.87 / 0.908, 0.961 |
| 2 | 38.20 / 37.23 | 40.00 / 38.00 | 0.90, 0.93 / 0.934, 0.961 |
| 3 | 39.65 / 38.44 | 40.04 / 38.10 | 0.92, 0.93 / 0.955, 0.961 |

Within ≈1.6–1.9 dB of Table I at 2–3 neurons, 3–4 dB *above* it at 1 neuron; prototype > classic
at 1–2 neurons and on every windowed SSIM (the paper's qualitative claim), classic wins at 3.
Eq 8 taken literally (global statistics) saturates at 0.999 for every model, so the paper's
0.87–0.93 must be a windowed SSIM.  Encoding ablation: θ = π·s loses 0.6–7.2 dB everywhere.
MIT-BIH record 100: CQNN 37.5/37.7/37.9, PTQNN 36.2/36.6/36.5 dB (input 33.9).  Fig 13 reproduced
(`fig13.png`).  **Table I: reproduced to within ~2 dB with ordering partly reproduced; exact
numbers depend on the unknown recording and noise level.**
