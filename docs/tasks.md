# Tasks

Each task module states what the paper fixes (quoted, pinned), what it leaves open (marked
`[ASSUMPTION]`), and exposes a loss for `kwsa.minimise`. Numbers below are from `STATE.md`;
the run directories they cite are under `experiments/runs/` (gitignored; regenerate with the
scripts in `experiments/README.md`).

## Bit-count (`tasks/bitcount.py`) — `[PTQNN §III.C, Fig 3–6]`

* Network: 5 inputs, 5 prototype hidden (Fig 5; the text says 6, Fig 4), 1 classic output —
  11 qubits, 36 parameters. `network(n_bits, n_hidden, hidden)`.
* Target: Fig 5's histogram read off as `p(x) = (popcount(x)+1)/112` (±0.001 per bar; the `+1`
  keeps `00000` representable). Confirmed on all 32 bars to ±0.0002.
* Superposition training: `superposition_prep` = `H` on inputs; `kl_loss` =
  `KL(target ‖ P(x | out=1))`; `train_superposition`. Median KL < 0.005 at sweep 9 (paper ≈ 8);
  5/6 seeds < 1e-3 at 20; seed 2 stalls at 0.014.
* Iterative baseline (Fig 6, red): `basis_prep(net, x)`, `sample_loss` (squared error of
  `P(out=1)` against `scale·(popcount+1)/(n+1)`), `train_iterative` — one KW–SA step per basis
  state. Stalls at ≈ 0.15 after 200 iterations where the paper reaches ≈ 0 by 120. **Recorded
  discrepancy**: the paper does not define its iterative scheme; ours is an assumption.
* Fig 3: single prototype neuron, prototype `(π, 0)`, `θ = π/2`, inputs `RX(πq)`: corners
  1.000 / 0.000 / 0.5 / 0.5 — reproduced (the paper's surface bottoms at ≈ 0.2 with unstated
  parameters).
* Scripts: `bitcount_figs.py`, `bitcount_noise.py`, `bitcount_shots.py`.

## Iris (`tasks/iris/`) — `[PTQNN §III.C, Fig 7–9]`, `[SNN §IV.B]`

* `data.py`: stratified seeded split (`load_split(seed, train_size)`), per-feature 1-D k-means
  `k = 3` fitted on the training rows, 12-bit one-hot-per-feature code (bit `3f + c`, MSB =
  "feature 0, cluster 0"), duplicate codes merged with the mean label `[ASSUMPTION]`,
  `encoding_accuracy_ceiling` (best any function of the code can do: 97/99/99 % on our splits).
  The split itself is an assumption — Fig 8's bin counts do not fix it.  Ambiguous codes
  (one 12-bit pattern carrying two class labels) follow `[Rosenhahn e-mail 2026-08-21]` ("I
  removed binary patterns which were assigned to two classes"): `load_split(...,
  ambiguous="drop")` removes every such code from both splits, while the default "merge"
  averages the labels and keeps all existing numbers unchanged; the experiment script exposes
  this as `--ambiguous {merge,drop}` (a "drop" run directory gets the suffix `-drop`).
* `model.py`: `dataset_superposition` loads the uniform superposition over the *distinct* codes
  as `Circuit.initial_state` (exact state prep; equivalent to Möttönen/QRAM noiselessly); target
  `p(x) ∝ (label+1)²` — Fig 8's bar heights confirm the square; `fast_joint` post-selects
  `out = 1` without simulating the output qubit; `predict_labels` = nearest level after a
  label-free scale fit; `accuracy` per sample.
* `train.py`: `IrisConfig` (4 hidden `[SNN Table I]` → 17 qubits; init ±0.7 rad to match
  Fig 7's starting KL), `run_fig7` trains classic and prototype nets of equal parameter count,
  `tsne_embedding` for Fig 9.  Because a t-SNE plot is not evidence (`[session-4 analysis]`,
  cf. the co-author's request for "another metric"), `train.prototype_placement` reduces Fig 9's
  claim to numbers: the hidden layer is decoded to one-hot patterns and scored by the covering
  cost `R` — the mean halved Hamming distance (one wrong block = 2 bits) from each distinct
  training code to its nearest pattern — relative to `covering_null` (random `k`-subsets of the
  81 valid patterns) and `covering_floor` (the best `k`-subset of the codes themselves), giving
  `rho = (R_null − R)/(R_null − R_floor)` with 1 = "as good as the best `k` medoids" and
  0 = "no better than random patterns".  It is computed for prototype and classic in both
  encodings ("inverse" = the Fig 9 cos²/sin² map, "direct" = the complementary sin²/cos²) on
  the training codes and written to `results.json["placement"]`; dead neurons (negligible
  output coupling) are excluded from the pattern set.  Fig 9's claim counts as reproduced only
  if, over ≥ 20 (seed, split) pairs, median rho_prototype ≥ 0.5 with a bootstrap CI excluding
  0 and rho_prototype > rho_classic (paired one-sided Wilcoxon, p < 0.01).
* Results: accuracy 95/91/98 % (three seeds); Fig 7 contrast reproduced (classic test KL rises,
  prototype monotone); **Fig 9 not reproduced** (prototype weights do not visibly span the data
  in TSNE). Script: `iris_fig7_fig8_fig9.py` (two 17-q statevector processes, `OMP_NUM_THREADS=1`).
* Noise: 17 q is beyond density matrices → trajectories, reduced run only
  (`ecg_iris_noise.py` study B); state preparation cost is the open hardware question
  (`reports/hardware-plan.md §3.3`).

## Flappy Bird (`tasks/flappy/`) — `[PTQNN §III.D, Fig 10, 11]`

The paper publishes the circuit of the trained agent (Fig 10: `RX(1.9537)`, `RX(2.1962)`,
four prototype neurons, output bias `RX(−1.4277)` and four CRX weights) and a runtime claim
(score 450 000, 6 days). It publishes no game, physics, reward or protocol.

* `env.py`: deterministic headless clone, numpy only. `FlappyConfig` holds geometry (288×512,
  gap 120, spacing 180), physics (gravity 1, flap −7, scroll 4), reward (0.01/frame, 1/pipe);
  all `[ASSUMPTION]`. Observations: `features_bird_gap` (bird height, next gap centre —
  the two boxes Fig 10 highlights); `features_bird_gap_distance` is an ablation.
* `agent.py`: `angles` maps a feature `x` to `θ = 2·asin(√x)` so `P(1) = x` (pinned by
  `[SNN Fig 4]`: `CRX(1.369) ↔ 0.4`); `QuantumFlappyPolicy` flaps iff `P(out=1) > ½`
  (`Network(2, 4)`, 7 qubits, 17 parameters), caches decisions on a quantised grid, records
  per-frame margins, optionally decides from shots or an `SPRTDecision`; `ThresholdPolicy` is the
  bang-bang baseline proving the clone is solvable; `render_*`/`save_gif` for Fig 11.
* `train.py`: loss = `−mean shaped return + λ·tracking error` over a few seeded episodes (the
  raw return is nearly binary — die at ~30 frames or survive — so finite differences of it are
  zero almost everywhere); 300 random restarts then 25 KW–SA sweeps; optional `SeedRotation`
  (off by default; needs `restate_incumbent`). Budget: 5000-frame cap = 110 pipes.
* Results: 110/110 pipes on 20/20 held-out seeds, every optimiser seed tried. Under noise and
  shots: `flappy_noise.py`, `flappy_shots.py` (margins 0.02 → ≳ 10⁴ shots/frame for fixed-N
  perfect play; SPRT ÷4.6). Scripts: `flappy_train.py` (writes `params.npy`, `log.csv`,
  `best.gif`).

## ECG denoising (`tasks/ecg/`) — `[PTQNN §III.D, Fig 12, 13, Table I, Eq 6–8]`

The paper names no recording, sampling rate or noise level.

* `data.py`: `synthetic_ecg` (sum-of-Gaussians PQRST beat model tuned to the landmarks measured
  off Fig 13's raster: 1800 samples, range 0.117–0.903, beats at t ≈ 370 and 1269, σ ≈ 0.03–0.04)
  and `load_physionet` (MIT-BIH record 100 via `wfdb`, cached in `data/raw/mitdb/`); `add_noise`
  (additive Gaussian, `DEFAULT_SIGMA`); `windows`/`supervised_windows` (5 noisy samples
  `s(t−2..t+2)` → clean `s(t)`); `contiguous_split` with a guard band. `load_mailed` reads the
  co-author's *original* recording (`data/raw/ecg-rosenhahn/*.mat`, `[Rosenhahn e-mail
  2026-08-21]`, `[ASSUMPTION]` that this is the Fig-13 signal — its 1800-sample range matches),
  which arrives pre-noised so `add_noise` is not applied and σ is measured (≈ 0.035),
  `mailed_provenance` pins the files by SHA256 for `meta.json`.
* `model.py`: `encode` = one `RX(θ_i)` per input with the same `θ = 2·asin(√s)` map;
  `predict` — `ŝ(t) = P(out=1)`, nothing rescaled (Fig 12 wires the output straight to the
  trace); `psnr`/`ssim` implement Eq 6–8 literally, with `ssim_windowed` as the scikit-image
  convention; `mse_loss` over a fixed random 600-window subsample.
* `train.py`: `prepare` → `EcgDataset`; `train` one cell; `table1` the 2×3 grid
  (CQNN/PTQNN × 1/2/3 hidden); `best_of` selects the restart by *training* MSE; `PAPER_TABLE1`
  holds the paper's numbers verbatim.
* Results (synthetic): PSNR 37.4 / 38.0 / 38.1 dB for 1/2/3 prototypes vs paper 32.9 / 40.00 /
  40.04 — within ~2 dB, Fig 13 reproduced; seed-to-seed spread of a cell is large (25–37 dB),
  hence restarts. Script: `ecg_table1.py` (one process per cell).

## Shared conventions

* Real-valued features → `RX(2·asin(√s))` everywhere (Flappy, ECG); `θ = π·s` is the ablation
  (`encoding=` / `mapping=` arguments).
* Every task's fast path (`backends.batched`) is pinned to the `Network.circuit` + reference
  route in its test file; the noisy evaluation of the same task goes through `NoisyBackend`.
* Prototype and classic comparisons are always at equal `n_params` (same shape).
