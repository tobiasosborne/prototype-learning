# Brief pi-06 — code fixes from Codex review 004

You are a coding agent in /home/tobias/Projects/prototype-learning (uv venv, Python 3.12). Work ONLY
in: `src/ptqnn/train/kwsa.py`, `src/ptqnn/tasks/bitcount.py`, `src/ptqnn/tasks/iris/train.py`,
`src/ptqnn/tasks/iris/data.py`, `experiments/bitcount_figs.py`, `experiments/ecg_table1.py`,
`experiments/iris_fig7_fig8_fig9.py`, `experiments/report_figures.py`, `tests/test_bitcount.py`,
`tests/test_train.py`, `tests/test_iris.py`. Do NOT commit; do NOT run experiments/*.py end to end
(unit tests only; each < 30 s). Do not edit any .tex file. Read `reviews/004-inbox.md` first (the
review; item numbers below refer to it). Red → green: for each behavioural change, a failing test
first. Gates at the end: `uv run ruff check src tests experiments && uv run ruff format src tests
experiments && uv run mypy src && uv run pytest tests/test_bitcount.py tests/test_train.py
tests/test_iris.py -q -m "not slow"`.

1. (Blocker 2) `experiments/bitcount_figs.py`: store BOTH `iterative_loss` (best monitored, as now)
   and a new `iterative_final` (last history value) per seed; `final_kl` in meta.json and the
   "final KL per seed" table in results.md must use the FINAL value, and add a "best" column.
   Same for the superposition series if it has the same ambiguity (check `res.loss` vs
   `history[-1]`). Also save the trained parameters of both schemes per seed
   (`params-superposition-<seed>.npy`, `params-iterative-<seed>.npy`) so diagnostics can be
   recomputed later; record `hostname` (socket.gethostname()), `platform.platform()`,
   `os.cpu_count()`, and the pool size in meta.json.
2. (Major 2) `bitcount_figs.py` plot: the monitor records an extra off-cadence final point when
   `iterations % monitor_every != 0`; place the last point at x = `iterative_iterations` (read from
   the results/meta), not `index × monitor_every`. Add a unit-level check by factoring the x-axis
   computation into a small function `monitor_x(n_points, monitor_every, iterations)` and testing it
   in `tests/test_bitcount.py` for (5 iterations, every 2) → [0, 2, 4, 5].
3. (Major 1) `bitcount.activation_distribution`: when every activation is zero return the ZERO
   vector (not uniform) so `perbasis_kl_loss` penalises a dead model through the same epsilon
   floor as `kl_loss`; update docstring and the test.
4. (Minor 1–3) `kwsa.py` docstring: the sole candidate is `x − a_n ĝ`. Validate `n_coords`
   (None or ≥ 1), and in `train_iterative` `anneal_every ≥ 1`, `batch` None or 1 ≤ batch ≤ 2**n;
   raise ValueError otherwise (tests). Reword "once per full pass over the basis inputs" to
   "once per nominal epoch of 2**n_inputs sampled inputs (with replacement)".
5. (Blocker 3) `experiments/ecg_table1.py`: replace every elapsed-time `time.time()` pair with
   `time.perf_counter()`; record `processes`, `hostname`, `platform` in meta.json.
6. (Major 5) `iris.train.dead_hidden_neurons`: when the strongest drive is 0 (or below `tol_abs`),
   ALL neurons are dead; test on zero parameters.
7. (Major 6) `firing_placement`: default `n_draws=1000`; the test that checks the null statistics
   must use ≥ 1000 draws with a stated tolerance (it costs ≈13 ms per draw — keep it under 30 s).
8. (Major 4) `tests/test_iris.py`: add a test that the fast route's FULL hidden-register density
   matrix (2^n_hidden × 2^n_hidden, obtained from `_hidden_layer_factors` or whatever internal the
   fast route exposes — expose a helper `hidden_register_state(net, params, code)` if needed)
   equals the reference backend's partial trace over the input qubits, to 1e-12, for random
   prototype AND classic parameters on the real 12→4 geometry. Marginals alone are not enough
   (the repo rule: test channels, not marginals). If the fast route does not hold the full state,
   say so and make it do so.
9. (Major 3) Iris drop policy: keep the current behaviour under `ambiguous="drop"` but ALSO keep the
   unfiltered test set: add fields `test_codes_all`, `test_labels_all` to `IrisSplit` (equal to the
   filtered ones under "merge"), filter `train_index`/`test_index` consistently with the codes
   (fix the broken invariant, test it), and make the experiment script report BOTH test accuracies
   (`test_accuracy` on the policy's test set and `test_accuracy_unfiltered` on all test samples)
   in results.json and the printed summary, also in `--rescore`.
10. (Major 7) `--rescore`: if the run dir has a `meta.json`, read `seed`, `train_size`,
    `ambiguous` from it and refuse to proceed when a CLI flag contradicts it; record in
    `meta-rescore.json` which values came from the original meta and which from the CLI; add
    `hostname`, `platform`, and wall time.
11. (Major 9) `experiments/report_figures.py::fig_bitcount_convergence`: until the workstation
    regenerates the figure, the left panel must read the same September run as the right one
    (`RUNS["bitcount_iterative"]`), with a comment `# TODO(workstation): switch back to
    RUNS["bitcount"] (six seeds) and restore "six seeds" in the caption`.

Finish by printing: files changed, the pytest summary line, and anything you could not do.
