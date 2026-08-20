# HANDOFF

End of session 1 (2026-08-20). Read STATE.md, then reviews/002-tasks-backends-hardware.md
"Resolution" — the open items there are the work queue, in this order:

1. Regenerate the numbers in reports/hardware-plan.md from the run data (blocker 1).
2. Promote the tasks' batched statevector evaluators into `backends/` (review 002 #7); then
   wire `restate_incumbent` into Flappy seed rotation (#8) and fix the Iris bin-count wording (#9).
3. Run the 11-qubit noise-aware bit-count training (≈1.3 h per cell on orkan @8 threads, or on
   the GPU now that it is fixed — re-measure; the script has progress.log and partial results):
   `experiments/bitcount_noise.py --studies B --b-network full`.
4. Integrate the Flappy-noise and ECG/Iris-noise results into reports/noise.md (run dirs under
   experiments/runs/flappy_noise, experiments/runs/ecg_iris_noise).
5. Review round 3 over the hardware fixes and the noise scripts.

Gates: `sh scripts/check-all.sh` (set LD_LIBRARY_PATH=/usr/lib/wsl/lib for GPU tests; `-m orkan`
takes minutes). Subagents: Opus/Sonnet only; Codex via docs/codex.md.

## Shot-noise programme (from the session-1 discussion; not yet implemented)

Context: the noise studies (`reports/noise.md`) show gate noise at calibrated vendor levels is
tolerated by the 7–11-qubit networks, but *shot noise* is the binding constraint — Flappy needs
≥ 10⁴ shots/frame with the plain binomial estimator because the trained decision margin
δ = |P(out=1) − ½| has median 0.02 (σ = 1/(2√N)); bit-count's plug-in KL is biased by
(K−1)/(2N·P(out=1)) and discards the `out = 0` half of the shots.  Both figures are ceilings
for the naive estimator, not requirements.  Ideas, roughly in order of payoff / cost:

1. **Margin-aware training (cheapest, do first).**  Add a hinge term to the Flappy objective
   (and the ECG/Iris activation objectives): penalise visited states with |P(out=1) − ½| < m,
   m ≈ 0.1.  Costs nothing noiselessly; shots scale as 1/δ², so 0.02 → 0.1 is ×25 fewer.
   Where: `tasks/flappy/train.py` objective; report margin quantiles as in
   `experiments/flappy_noise.py`.
2. **Common random numbers for KW–SA on sampled backends.**  Evaluate the ±c probes of each
   coordinate with the *same* shot seed (on hardware: interleave the ± circuits in one job).
   The finite-difference variance drops from O(1/N) to O(c²/N).  Two-line change: a
   `ShotBackend(inner, shots, seed)` that samples `reference.sample_counts` and a seed shared
   per probe pair in `kwsa.kw_gradient` / the coordinate loop.  This is what makes
   finite-difference optimisation usable on a sampling backend at all.
3. **Sequential testing for decisions (Flappy).**  Replace fixed-N binomial with a sequential
   probability ratio test on the sign of P − ½: far-from-boundary states resolve in tens of
   shots, only the ~10 % near-boundary frames pay the full budget.  Expected 5–20× on mean
   shots/frame.  Where: `tasks/flappy/agent.py` `act()` with a `ShotPolicy`.
4. **Do not discard shots (bit-count / Iris).**  Either train on the joint P(x, out) with
   target (popcount+1)/112 · P(out=1) (normaliser from all shots), or apply one Grover/amplitude
   amplification step on the `out = 1` branch (≈ doubles the usable fraction; +1 circuit depth).
   Where: `train/distribution.py` (joint variant), `circuits/` (amplification op).
5. **Better divergence estimators.**  Miller–Madow or Dirichlet-smoothed KL; or switch the
   training loss to a bounded proxy with unbiased finite-sample estimators — the paper's own
   cosine dissimilarity (Eq 5) or total variation.  Where: `train/losses.py`; quantify bias vs
   shots with the existing study-D harness in `experiments/bitcount_noise.py`.
6. **Amplitude estimation.**  P(out=1) is an amplitude² of a unitary circuit → iterative /
   maximum-likelihood QAE gives error ∝ 1/N_oracle (Heisenberg) instead of 1/√N: δ = 0.02 needs
   ~50 oracle calls × small constant instead of 2500 shots.  Cost: 2–6 repetitions of the
   circuit in depth — fine on all-to-all trapped ions, not on heavy-hex.  Where: new
   `hardware/qae.py`; simulate first with the statevector backend.
7. Classical shadows are not useful here (single-qubit Z observable).

Suggested order: (1) + (2) → rerun `experiments/flappy_noise.py` study C and a sampled-backend
bit-count training; then (3), (4); (5) as a side study; (6) only for the hardware-plan appendix.
Deliverable: a "shots" section in `reports/noise.md` with shots-to-target curves before/after.
