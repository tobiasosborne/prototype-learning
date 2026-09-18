# Brief pi-03 — bit-count iterative baseline: right observable, right iteration unit

You are a coding agent in the Python repo /home/tobias/Projects/prototype-learning (package
`src/ptqnn`, uv venv, Python 3.12). Work ONLY in: `src/ptqnn/train/kwsa.py`,
`src/ptqnn/tasks/bitcount.py`, `experiments/bitcount_figs.py`, `tests/test_train.py`,
`tests/test_bitcount.py`. Do NOT commit. Do NOT touch other files. Do NOT run
`experiments/bitcount_figs.py` (it is slow); unit tests only, each under ~30 s.

## Background (facts established this session; cite them as `[session-4 analysis]`)
The bit-count task (5 input qubits, 5 prototype hidden neurons, 1 output; `bitcount.network`)
is trained two ways. `train_superposition` minimises `kl_loss` = KL(target ‖ model) where the
model distribution is P(inputs = x, out = 1) renormalised over x, measured on the uniform
superposition of inputs `[PTQNN Eq 4]`. `train_iterative` (the Fig 6 "red" baseline) takes
one KW–SA step on the squared error of ONE random basis input per iteration `[SNN §III.A]`.
Three defects were found in `train_iterative`:
(a) It is monitored with `kl_loss` (bitcount.py ~l.127), the superposition observable. But the
    prototype block RX(φ)·CRX·RX(−φ) does not restore the input register (the un-compute acts on
    a control that the CRX has entangled), so P(inputs=x, out=1) ≠ 2^-n · P(out=1 | x). A network
    fitted to the per-input activations has per-input KL ≈ 4e-4 while its superposition KL is
    ≈ 0.03–0.09, and vice versa. The sample-wise scheme must be graded on the per-input
    distribution: q(x) ∝ P(out=1 | x) over the 32 basis inputs, target (popcount(x)+1)/112.
(b) `kwsa.minimise` (kwsa.py ~l.152) accepts the ±c probes themselves as candidates. On a
    one-example objective that is a greedy random walk (KL history 0.155→0.271→0.093→…→0.344).
(c) `train_iterative` passes `start_iteration=n` per EXAMPLE, so T = t0·cooling^n underflows by
    n≈50 and c_n never shrinks; and it reuses `SUPERPOSITION_CONFIG` (a0=20 is ~10× too large
    for single-example steps). The paper's "iteration" for the red curve is most plausibly one
    elementary KW step on one coordinate on one example (the author says to use "1000 times more
    iterations"); our iteration is a 36-coordinate sweep = 109 evaluations.

## Repo rules
- Pin facts in docstrings (`[PTQNN Eq 4]`, `[SNN §III.A]`, `[session-4 analysis]`); our choices
  are `[ASSUMPTION]`. Keep existing behaviour and numbers of `train_superposition` and of
  `kwsa.minimise` with default config EXACTLY unchanged (existing tests must pass untouched).
- Red → green: write the failing tests first, run them, then implement.
- Gates: `uv run ruff check src tests experiments && uv run ruff format src tests experiments
  && uv run mypy src && uv run pytest tests/test_train.py tests/test_bitcount.py -q`.
- Read kwsa.py and bitcount.py fully first; mirror style (frozen dataclasses, type hints).

## Deliverable
1. `kwsa.py`, `KWSAConfig`: two new fields with defaults that reproduce today's behaviour:
   `n_coords: int | None = None` (coordinate mode visits only the first `n_coords` of the random
   permutation per iteration; None = all) and `probe_candidates: bool = True` (when False, only
   the KW step `x + step` is a candidate; the ±c probes are used for the derivative only).
   Docstrings explain (b) above. Also add `anneal_every` handling? NO — keep kwsa generic.
2. `bitcount.py`:
   - `activation_distribution(net, params, backend=None) -> Real`: the 32 activations
     P(out=1 | x) via `basis_prep`, renormalised to sum 1 (guard the all-zero case by returning
     the uniform distribution).
   - `perbasis_kl_loss(net, backend=None) -> Loss`: KL(target_distribution ‖ activation_distribution),
     same KL convention/epsilon handling as `kl_loss`.
   - `ITERATIVE_CONFIG = kwsa.KWSAConfig(iterations=1, a0=2.0, alpha=0.0, c0=0.05, gamma=0.0,
     t0=0.0, n_coords=1, probe_candidates=False)` with a docstring `[ASSUMPTION]` + the
     rationale from (b)/(c).
   - `train_iterative(..., config=None, monitor: Literal["perbasis","superposition"]="perbasis",
     monitor_every: int = 1, anneal_every: int | None = None, batch: int | None = None)`:
     default config becomes `ITERATIVE_CONFIG`; `start_iteration = n // anneal_every` with
     `anneal_every` defaulting to `2**net.n_inputs`; the KL is recorded every `monitor_every`
     iterations (history index i ↔ iteration i·monitor_every; document) and ALWAYS at the end;
     record BOTH series: `KWSAResult.history` holds the chosen monitor, and put the other one in
     a new optional field `KWSAResult.extra_history: dict[str, list[float]]` (default empty)
     under key "superposition" or "perbasis". `batch=k` (k ≤ 32) means the per-iteration
     objective is the mean squared error over `k` random distinct basis inputs (k = 32 → full
     batch); `batch=None` keeps the single-example scheme. Return type stays `KWSAResult`.
3. `experiments/bitcount_figs.py`: flags `--iterative-monitor {perbasis,superposition}`
   (default perbasis), `--iterative-batch INT` (default 0 = single example), `--iterative-
   monitor-every INT` (default 1); pass them through the worker; write them into `meta.json`;
   plot the chosen monitor and, if present, the other series dashed; keep all defaults so the
   existing superposition curve and file names are unchanged. Do not run it.
4. Tests:
   - `tests/test_train.py`: with `probe_candidates=False` the accepted point after one iteration
     differs from `x0` only along the KW step direction (on a quadratic f, check the update equals
     −a·grad exactly to 1e-9 for `n_coords=1`); `n_coords=1` performs exactly 3 evaluations per
     iteration; defaults reproduce the previous result bit-for-bit (compare a run with explicit
     `n_coords=None, probe_candidates=True` against one with the old defaults on a fixed seed).
   - `tests/test_bitcount.py`: `activation_distribution` sums to 1 and equals
     P(out=1|x)/Σ for random params (cross-check against `basis_prep` + the numpy backend
     directly); the superposition and per-basis distributions DIFFER for random prototype params
     (max abs gap > 0.01 — this is claim (a), make it a test); `perbasis_kl_loss` is 0 to 1e-12
     when `activation_distribution` is patched to equal the target; `train_iterative(net, seed=0,
     iterations=64, monitor_every=16)` returns a history of length 5 with both series present;
     `batch=32` runs and its history is finite; `anneal_every` default equals 32.
5. Keep the per-iteration cost low: with `n_coords=1` an iteration must cost exactly 3 loss
   evaluations plus at most one monitor evaluation every `monitor_every`.

Finish by printing: files changed, the pytest summary line, and any `[ASSUMPTION]` beyond
those above.
