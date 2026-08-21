# HANDOFF

End of session 2 (2026-08-21). Read STATE.md first. The deliverable is `reports/tex/main.pdf`
(build: `cd reports/tex && latexmk -pdf main.tex`; figures: `experiments/report_figures.py`).

## Work queue, in order

1. **Review round 3.** `reviews/003-session2.md` holds the Codex (gpt-5.6-sol, xhigh) adversarial
   review of the session-2 diff (`git diff 7271620..HEAD`). Apply or explicitly decline each
   finding in a "Resolution" table at the bottom, re-run `sh scripts/check-all.sh`, and update
   the tex report if a number changes (the report quotes run dirs only — regenerate, do not edit).
2. **Shot-noise programme item 4** (the one that now limits sampled training): a loss that uses
   all shots — train on the joint P(x, out) with target (popcount+1)/112 · P(out=1), or one
   amplitude-amplification step on the `out=1` branch. Where: `train/distribution.py`
   (joint variant), `circuits/` (amplification op). Measure with `experiments/bitcount_shots.py`
   (add a `--loss joint` flag); the plug-in bias (K−1)/(2·P(out=1)·N) is the number to beat.
3. **Item 5**: Miller–Madow / Dirichlet-smoothed KL, or the paper's cosine loss (Eq 5) with an
   unbiased estimator, in `train/losses.py`; same harness.
4. **Item 6** (hardware appendix only): iterative/ML amplitude estimation of P(out=1) —
   `hardware/qae.py`, simulate on the statevector backend first.
5. Optional, budget permitting (each ≤ 1 h): one cell of 11-q noise-aware bit-count training
   (`experiments/bitcount_noise.py --studies B --b-network full`, ibm_fez, prototype, 10 sweeps,
   GPU ≈ 30 min); a fuller ECG noise block (`experiments/ecg_iris_noise.py`, 100 windows,
   2 presets ≈ 15 min). Ask before anything longer.

## Rules that bit us this session
* No run > 1–2 h; estimate from a measured per-eval cost first and print it (`flush=True`).
* Penalties in a shaped objective must be priced below the per-frame reward (the mean-hinge
  attempt produced a confidently dead Flappy policy).
* Non-stationary objectives need `KWSAConfig.restate_incumbent=True` (now automatic for
  `flappy.train(batch>0)`).
* Common random numbers for finite differences need the *maximal coupling*
  (`backends/shots.py`), not shared inverse-CDF uniforms (1.3× only).
* Every hardware number goes through `experiments/hardware_plan.py`; never hand-edit the report.

Gates: `LD_LIBRARY_PATH=/usr/lib/wsl/lib sh scripts/check-all.sh` (green at session close).
Subagents: Codex only this session (`docs/codex.md`); no Claude subagents were used.
