# Brief pi-05 — report figure `fig_bitcount_convergence`: the iterative baseline in its own observable

You are a coding agent in /home/tobias/Projects/prototype-learning (uv venv, Python 3.12).
Work ONLY in `experiments/report_figures.py`. Do NOT commit. Do not touch other files.

## Context
`experiments/report_figures.py` regenerates every figure of `reports/tex/` from pinned run
directories (`RUNS` dict near the top; `save(fig, name)` writes `reports/tex/figures/<name>.png`
— read the module header, `load`, `save`, and the colour constants PROTO/ORANGE/INK/INK2/GRID).
`fig_bitcount_convergence()` (≈ line 200) currently draws two panels from
`RUNS["bitcount"]/histories.json`: superposition training (left) and the OLD iterative baseline
(right, annotated "ours stalls ≈0.15").

New facts (`[session-4 analysis]`): the old right panel graded the sample-wise scheme with the
superposition KL, a different observable from the per-input activation distribution it trains
on. A new run, `experiments/runs/bitcount/20260918T111515Z`, trains the baseline with one
elementary Kiefer–Wolfowitz step (one coordinate, one random input) per iteration, 100 000
iterations, 4 seeds, and records the per-input KL every 1000 iterations. Its `histories.json`
has keys: `"superposition"` (dict seed → list of 21 KLs, one per sweep), `"iterative"` (dict
seed → list of 101 per-input KLs, index i ↔ iteration 1000·i), `"iterative_monitor"` (the
string "perbasis"), `"iterative_superposition"` (dict seed → list of 101 superposition KLs of
the same networks). Results: seeds 1/2/3 reach per-input KL < 1e-3 at 73k/23k/12k iterations,
seed 0 plateaus at 0.02–0.03; the superposition KL of these networks stays 0.07–0.4. The paper's
Fig 6 red curve reaches ≈1e-3 by "≈120 iterations" — its iteration unit is not stated
(`[ASSUMPTION]`: one elementary step, so 120 plotted iterations ≈ 1e5 steps is plausible).

The OLD run directory `RUNS["bitcount"]` (20260820T135930Z) is NOT present in this checkout
(it lives on the workstation); the NEW one is. Do not change what the left panel reads.

## Deliverable
1. Add `"bitcount_iterative": ROOT / "experiments/runs/bitcount/20260918T111515Z"` to `RUNS`.
2. Rewrite `fig_bitcount_convergence()`:
   - left panel unchanged in content (superposition sweeps from `RUNS["bitcount"]`, log-y,
     paper annotation);
   - right panel from `RUNS["bitcount_iterative"]`: x = iteration (index × 1000, read the step
     from the data length: 100 000 / (len − 1)), per-seed per-input KL in ORANGE (alpha 0.55),
     median in INK, log-y (same shared y as the left), the `iterative_superposition` median as
     a thin dashed INK2 line labelled "superposition KL of the same network", a horizontal
     GRID line at 1e-3, x on a log scale starting at 1000, title
     "iterative (one KW step on one input per iteration)", xlabel "iteration",
     annotation "paper: ≈1e-3 after ~120 iterations (unit unstated)" placed near the 1e-3 line;
   - update the caption-like `fig.text` at the bottom: "left: one sweep = 3 × 36 circuit
     evaluations, 6 seeds; right: one iteration = 3 evaluations, 4 seeds; the KL on the right is
     over the 32 per-input activations, the paper's Eq. 4 KL of the same networks is dashed".
   - keep figsize (6.6, 2.8) and the file name `fig_bitcount_convergence`.
3. Because `RUNS["bitcount"]` is absent here, test with a harness, not the CLI: in a
   `uv run python - <<EOF` snippet, `import sys; sys.path.insert(0, "experiments")`, import the
   module, set `report_figures.RUNS["bitcount"] = report_figures.RUNS["bitcount_iterative"]`
   (the new run also has a `superposition` key, 4 seeds — fine for the smoke test), call
   `fig_bitcount_convergence()`, and confirm `reports/tex/figures/fig_bitcount_convergence.png`
   is written with no exception. Then `git checkout -- reports/tex/figures/fig_bitcount_convergence.png`
   so the committed figure (from the workstation's 6-seed left panel) is not replaced by the
   smoke-test output; say explicitly that you did this. Also make sure that the function does
   not hard-code "6 seeds" — derive the count from the data.
4. Gates: `uv run ruff check experiments && uv run ruff format experiments && uv run mypy src`.

Finish by printing the diff summary and the exact test snippet you ran.
