# Brief pi-02 — Iris: "drop ambiguous codes" policy (co-author's actual preprocessing)

You are a coding agent working in the Python repo at /home/tobias/Projects/prototype-learning
(package `src/ptqnn`, uv-managed venv, Python 3.12). Work ONLY in these files:
`src/ptqnn/tasks/iris/data.py`, `src/ptqnn/tasks/iris/train.py` (only if a signature must be
threaded through), `experiments/iris_fig7_fig8_fig9.py`, `tests/test_iris.py`, `docs/tasks.md`
(Iris paragraph only). Do NOT commit. Do NOT touch any other file. Nothing you run may take
more than ~2 minutes (do not run the full Iris training; unit tests only).

## Context
The repo reproduces the Iris experiment of "Prototype Learning for Quantum Neural Networks"
(§III.C, Fig 7–9): each of the 4 features is k-means-binarised into 3 bits, giving a 12-bit
code per sample; the dataset is loaded as a superposition over codes with an amplitude-encoded
label. Several samples share a 12-bit code; today `data.merge_duplicate_codes` merges them and
AVERAGES the labels (an `[ASSUMPTION]`, since the paper is silent). The paper's author wrote on
2026-08-21: "On the iris-dataset I should have mentioned that I removed binary patterns which
were assigned to two classes." That is a different policy: drop every code that occurs with
more than one distinct label; codes duplicated within one class are still merged (one basis
state, one amplitude).

## Repo rules you must follow
- Pin every fact: this policy is pinned as `[Rosenhahn e-mail 2026-08-21]`; anything the e-mail
  does not fix (e.g. whether the test set is filtered too) is `[ASSUMPTION]` and explained.
- Red → green TDD: write failing tests first in `tests/test_iris.py`, see them fail, then implement.
- Gates before you finish: `uv run ruff check src tests experiments && uv run ruff format src
  tests experiments && uv run mypy src && uv run pytest tests/test_iris.py -q`.
- Read `src/ptqnn/tasks/iris/data.py` fully (esp. `merge_duplicate_codes`, `MergedCodes`,
  `IrisSplit`, `load_split`, `encoding_accuracy_ceiling`) and how `train.py` and the experiment
  script consume them, before editing. Mirror the existing style and type hints.
- The DEFAULT behaviour must stay exactly as today (`"merge"`), so existing numbers do not move.

## Deliverable
1. `data.py`:
   - `AmbiguousPolicy = Literal["merge", "drop"]`.
   - `drop_ambiguous_codes(codes, labels) -> tuple[Codes, Codes, int]`: remove every sample whose
     code occurs with ≥ 2 distinct integer labels; return the kept codes, kept labels and the
     number of DISTINCT codes dropped. Pure numpy, deterministic, order-preserving.
   - Thread a keyword `ambiguous: AmbiguousPolicy = "merge"` through `load_split` (and
     `IrisSplit` as a recorded field, plus `n_dropped_train` / `n_dropped_test` ints, both 0 for
     "merge"). With `"drop"`, filter BOTH the train and the test codes ( `[ASSUMPTION]`: an
     ambiguous pattern cannot be scored either; state this in the docstring), then proceed to the
     existing merge of same-class duplicates. `IrisSplit.summary()` must include the policy and
     the two drop counts.
   - `encoding_accuracy_ceiling` on a dropped set must be 1.0 by construction — add an assertion
     or a test, not both.
2. `experiments/iris_fig7_fig8_fig9.py`: `--ambiguous {merge,drop}` (default merge), passed to
   `load_split`, recorded in `meta.json`, and appended to the run directory name suffix
   (e.g. `…-drop`) so runs do not overwrite each other. Do NOT run the script.
3. Tests (new, in `tests/test_iris.py`): a synthetic case with codes `[5, 5, 7, 7, 9]` and labels
   `[0, 1, 2, 2, 1]` → code 5 dropped (n_dropped=1), codes 7,7,9 kept in order; `load_split(seed=0,
   ambiguous="drop")` has zero collisions across classes and `encoding_accuracy_ceiling == 1.0`
   on train and on test; `load_split(seed=0)` is bit-for-bit identical to before (compare to
   `ambiguous="merge"` explicitly).
4. `docs/tasks.md` Iris section: two sentences on the new policy, its pin, and the flag.

Finish by printing: files changed, the pytest summary line, and any `[ASSUMPTION]` you made
beyond those listed.
