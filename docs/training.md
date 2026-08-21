# Training

One optimiser (`train/kwsa.py`), three losses (`train/losses.py`), and the observables that turn
an outcome distribution into the paper's model distribution (`train/distribution.py`). Every task
hands `kwsa.minimise` a closure `loss(params) -> float` built over a `Backend`.

## Kiefer–Wolfowitz inside simulated annealing (`kwsa.py`)

The paper `[PTQNN §II.C, Eq 3; SNN §III.A]` fixes only the shape: a central-difference gradient
estimate `(M(x + c_n) − M(x − c_n)) / 2c_n` proposes a neighbour, a Metropolis rule
`exp(−Δe/kT)` accepts it; no gradient descent, no autodiff. Neither paper publishes `a_n`,
`c_n`, the temperature schedule or `k`. Our choices, all `[ASSUMPTION]`:

| `KWSAConfig` field | meaning | default |
|---|---|---|
| `a0, alpha` | step `a_n = a0 / (n+1)^alpha` | 0.5, 0.602 (Spall's exponents) |
| `c0, gamma` | probe `c_n = c0 / (n+1)^gamma` | 0.1, 0.101 |
| `t0, cooling, k` | geometric `T_{t+1} = cooling · T_t` from `t0`; `k` damps Eq 3 | 0, 0.95, 1 |
| `mode` | `"coordinate"` (default) or `"full"` | coordinate |
| `jitter` | Gaussian exploration `jitter·T·N(0,1)` in full mode | 0 |
| `restate_incumbent` | re-score the incumbent every iteration (non-stationary objectives) | False |
| `iterations, seed` | | 100, 0 |

**Coordinate mode** (what every reproduction uses): one *iteration* = one sweep over all
coordinates in random order; per coordinate, evaluate `x ± c_n e_i`, propose `x − a_n ĝ_i e_i`,
and put the best of the three probes to the Metropolis rule. Cost `3·n_params` evaluations per
sweep (108 for the Fig-5 network). Whole-vector steps (`mode="full"`) stalled at KL ≈ 1e-2 at
every setting tried. The paper's "iteration" unit is undefined, so Fig 6 is reported against
both sweeps and evaluations (review 001 #8).

`KWSAResult`: `params`/`loss` are the best point seen (use these); `final_params`/`final_loss`
the annealer's last accepted state (resume from these, and *report* these under a rotating
mini-batch, where "best ever" only means "luckiest batch"); `history`, `best_history`,
`accepted`, `evaluations`. `minimise(f, x0, cfg, callback, start_iteration, map_fn,
probe_context)`: `map_fn` lets a caller evaluate the probes of one coordinate in parallel;
`probe_context` wraps exactly the `±c` and proposed-step evaluations (used for common random
numbers, below).

`restate_incumbent=True` is mandatory when the objective changes between iterations (rotating
episode seeds, mini-batches): with a stale incumbent loss the Metropolis comparison is against
the wrong number — Flappy held-out play went 20/25 → 0/25 perfect games without it and 25/25
with it (`experiments/runs/flappy/batch5`). `flappy.train(batch > 0)` switches it on.

### Per-task schedules

| task | config | notes |
|---|---|---|
| bit-count, superposition | `bitcount.SUPERPOSITION_CONFIG`: 20 sweeps, a0 20, α 0.3, c0 0.2, γ 0.101, t0 0.02, cooling 0.7 | median KL < 0.005 at sweep 9; paper ≈ 8 |
| bit-count, iterative | same, one step per basis state on the squared error | stalls at ≈ 0.15 — recorded discrepancy |
| Iris | `IrisConfig`: 40 sweeps, a0 2, α 0.3, c0 0.1, cooling 0.8, init ±0.7 rad | init scale chosen to match Fig 7's starting KL |
| Flappy | `flappy.train.DEFAULT_KWSA`: 25 sweeps, a0 0.4, c0 0.25, t0 0.05, cooling 0.9, after 300 random restarts | ≈ 1575 evaluations, ~3 min |
| ECG | `ecg.train.DEFAULT_CONFIG`: 100 sweeps, a0 200, α 0.3, c0 0.4, **γ 0.9**, greedy; 600-window subsample; best of 256 screened inits × restarts | γ is what matters: the probe step is the real move early on, so `c_n` must go coarse → fine |

Each is grid-searched only as far as the docstring says; none is claimed optimal.

## Losses (`losses.py`)

`kl_divergence(target, model, eps)` — `Σ p log(p/q)`, `0 log 0 = 0`, `q` floored at `eps`;
target first `[PTQNN Eq 4]`. `cosine_dissimilarity` — `1 − P·Q` on unit vectors `[PTQNN Eq 5]`
(implemented, not used for training). `squared_error` — the per-sample ℓ2 of the iterative
scheme. Flappy minimises a negative shaped episode return and ECG an MSE; those live in their
task modules.

## The superposition observable (`distribution.py`)

`[PTQNN Fig 5]`: Hadamards put all inputs in superposition; the histogram is indexed by the
*input* string and sums to one. So the model distribution is the joint `P(inputs = x, out = 1)`
renormalised over `x` — `conditioned_input_distribution` — **not** `P(out = 1 | x)`. Conditioning
on `out = 1` makes the output amplitude a weight on each input pattern ("an amplitude encoding
of the output", Fig 4 caption). `joint_input_output` returns the `(2^n_in, 2)` table;
`output_activation` is the sample-wise `P(out = 1)` for one prepared input (Fig 3, the iterative
baseline, Flappy, ECG). Iris uses the same observable with the dataset superposition as
`initial_state` and a target `∝ (label+1)²` (`tasks/iris/model.py`).

## Training on finite shots

Everything the paper trains is trained on exact probabilities. On hardware the loss is
`counts/N`, and three things go wrong, each addressed in the shot-noise programme
(`STATE.md`, `reports/noise.md` "Shots"):

1. **Gradient variance.** A central difference with independent shot noise has slope variance
   `O(1/(c²N))`. `ShotBackend.common_random_numbers()` + `kwsa.minimise(probe_context=…)`
   couples the probes maximally and brings this to `O(1/(cN))` — measured ÷2.9 at `c = 0.2`,
   ÷12–22 at `c = 0.05`. Variance reduction only; it does not remove bias.
2. **Estimator bias.** The plug-in KL of a sampled histogram is biased by about
   `(K−1)/(2·P(out=1)·N)` (`K = 32` bins, only ≈ 48 % of shots survive `out = 1`). Sampled
   bit-count training reaches 0.014 at N = 4000 against 0.0009 exact; consistent with the bias
   being the limit, untested beyond that. Queue item 4 (joint loss / amplitude amplification) and
   5 (Miller–Madow or Dirichlet-smoothed KL, or Eq 5 with an unbiased estimator) target it.
3. **Decision noise at deployment.** Flappy's flap decision `P(out=1) > ½` sits a median
   0.019 from the threshold, so a frame decided from `N` shots flips with probability
   `≈ Φ(−2δ√N)`. `flappy.agent.SPRTDecision` (Wald's test between `½ ± δ0`, blocks of 20,
   cap 10 000) keeps 100 % perfect play at 2 191 shots/frame vs 10 000 fixed. The indifference
   margin must sit below the margins that decide the game (p10 ≈ 0.005), not at the median.
   Margin-aware fine-tuning (per-frame hinge in the return) did not move the margins — negative
   result, kept.

Rule from the Flappy margin experiment: a penalty in a shaped objective must be priced below
the per-frame reward, or the optimiser learns to die early (`experiments/runs/flappy_shots/20260821T075048Z`).
