# Brief pi-04 — Iris: quantitative prototype-placement metric (replaces the Fig 9 t-SNE as evidence), plus the --ambiguous flag

You are a coding agent in /home/tobias/Projects/prototype-learning (package `src/ptqnn`, uv venv,
Python 3.12). Work ONLY in: `src/ptqnn/tasks/iris/train.py`, `experiments/iris_fig7_fig8_fig9.py`,
`tests/test_iris.py` (append new tests; keep the existing ones), `docs/tasks.md` (Iris section
only). Do NOT commit. Do NOT run the experiment script (slow). Unit tests only, each < 30 s.

## Background
Iris samples are 12-bit codes: 4 features × 3 k-means clusters, one-hot per feature, so every
valid code has exactly four 1-bits, one per 3-bit block (`data.code_bits`: (n, 12) uint8,
column 0 = MSB = feature 0 cluster 0). The network has 12 inputs, `n_hidden` hidden neurons
(default 4), one output. `train.hidden_layer_weights_for_tsne(net, params, as_pattern=True)`
returns (n_hidden, 12): cos²(φ/2) for prototype neurons (a large angle stresses a 0 input: the
"inverse" encoding `[Rosenhahn e-mail 2026-08-21]`), sin²(w/2) for classic. The paper's Fig 9
(t-SNE) claims prototype weights "span the data"; t-SNE is not evidence (`[session-4 analysis]`).
The co-author asked for "another metric" (e-mail 2026-08-27).

Also: `data.load_split(seed, train_size, ambiguous="merge"|"drop")` already exists (uncommitted,
pi-02), but `experiments/iris_fig7_fig8_fig9.py` does not expose it yet.

## Repo rules
Pin facts (`[PTQNN Fig 9]`, `[Rosenhahn e-mail 2026-08-21]`, `[session-4 analysis]`); own choices
`[ASSUMPTION]`. Red → green: failing tests first. Gates: `uv run ruff check src tests experiments
&& uv run ruff format src tests experiments && uv run mypy src && uv run pytest tests/test_iris.py -q`.
Read train.py, data.py and the experiment script fully first; mirror style.

## Deliverable A — metric (in `train.py`, next to `hidden_layer_weights_for_tsne`)
1. `decode_hidden_patterns(net, params, encoding: Literal["inverse","direct"]="inverse") ->
   npt.NDArray[np.uint8]` (n_hidden, 12): per neuron take m = cos²(φ/2) ("inverse") or sin²(φ/2)
   ("direct") for prototype, sin²(w/2)/cos²(w/2) respectively for classic; within each of the four
   3-bit blocks set the argmax to 1, others 0. Invariant under monotone rescaling — say so.
2. `dead_hidden_neurons(net, params, tol=0.05) -> npt.NDArray[np.bool_]`: a neuron whose output
   coupling is negligible, |sin(θ_j/2)·sin(w_j/2)| < tol where θ_j is the neuron's CRX angle and
   w_j its output-layer weight (read `Network.unpack` to find both; if the parameter layout makes
   one of them unavailable, use what exists and document it as `[ASSUMPTION]`).
3. `covering_cost(patterns: (k,12) uint8, codes: Codes) -> float`: with C the DISTINCT training
   codes (M of them), R = (1/M) Σ_{c∈C} min_j d_H(b_j, c)/2  (Hamming distance halved: one wrong
   block costs 2 bits). Range [0, 4].
4. `covering_null(k, n_draws=10_000, seed=0) -> tuple[float, float]`: mean and std of R over
   k-subsets drawn uniformly (without replacement) from the 81 valid one-hot patterns, against
   the same code set C — signature `covering_null(codes, k, n_draws, seed)`.
5. `covering_floor(codes, k) -> float`: min R over all k-subsets of C (brute force,
   `itertools.combinations`; M ≤ ~25 and k = 4 → ≤ 12 650 subsets; guard k > M).
6. `prototype_placement(net, params, codes, encoding="inverse", n_draws=10_000, seed=0,
   drop_dead=True) -> PlacementScore` dataclass with fields `R, R_null, R_null_std, R_floor,
   rho, n_dead, n_patterns, encoding` where rho = (R_null − R)/(R_null − R_floor); rho = 1 is
   "as good as the best k medoids", 0 "no better than random patterns"; when drop_dead, dead
   neurons are excluded from the pattern set (k shrinks; null and floor use the reduced k).
   Docstring: what counts as reproducing Fig 9's claim `[session-4 analysis]`: over ≥ 20
   (seed, split) pairs, median rho_prototype ≥ 0.5 with a bootstrap CI excluding 0, and
   rho_prototype > rho_classic (paired one-sided Wilcoxon, p < 0.01).

## Deliverable B — experiment script
- `--ambiguous {merge,drop}` (default merge) → `load_split(..., ambiguous=...)`; write it into
  `meta.json`; when "drop", suffix the run directory name with `-drop`.
- After training both kinds, compute `prototype_placement` for prototype (encoding inverse AND
  direct) and classic (inverse and direct) on the training codes; write all four `PlacementScore`s
  as dicts into `results.json` under `"placement"` and print one line per score. Keep Fig 9 as is.
- `figure9`'s replay path (l.~157 `load_split(seed=..., train_size=...)`) must pass the
  recorded `ambiguous` policy too (default merge if absent from an old meta.json).

## Tests (append to tests/test_iris.py)
- `decode_hidden_patterns` returns exactly one 1 per block; invariant under `params → 2·params`?
  NO (angles are not scale-free) — instead test invariance under any monotone map applied to the
  (n_hidden, 12) pattern matrix by calling the block-argmax helper directly on `m` and `m**3`.
- `covering_cost` of the codes against themselves (k = M) is 0; of one pattern at Hamming
  distance 2 from a single code is 1.0.
- `covering_floor(codes, k=M)` is 0; `covering_null` mean > floor on `load_split(seed=0)` codes.
- `prototype_placement` on random params returns finite fields and rho ≤ 1 + 1e-9.

## Deliverable C — docs/tasks.md (Iris section)
Two sentences on `ambiguous="drop"` (`[Rosenhahn e-mail 2026-08-21]`, flag `--ambiguous`) and
three on the placement metric (definition of R, null, floor, rho; that it replaces t-SNE as
evidence; `results.json["placement"]`).

Finish by printing: files changed, the pytest summary line, any `[ASSUMPTION]` beyond the above.
