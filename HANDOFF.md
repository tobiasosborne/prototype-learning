# HANDOFF

End of session 5 (2026-09-25, workstation). Read `STATE.md`, then the session-5 worklog entry.
Deliverables: `reports/tex/main.pdf`, `reports/tex/overleaf/` (drop-in for the co-author's
Overleaf; `standalone.pdf` builds it), `notes/reply-to-bodo-draft.md` (e-mail; fill [SLOTS]).

## State at close
* Review 004: every code finding closed (resolution table at the end of `reviews/004-inbox.md`).
* All laptop runs regenerated here from clean commits; where they overlap they are bit-identical.
  The laptop run dirs (`*/20260918T*`) exist only on the laptop; the report now cites the
  `20260925` runs (the compute paragraph names both).
* Iris 20-seed × {merge, drop} sweep done; Fig 9 verdict (pre-registered): **not reproduced** —
  prototype selectivity ≫ classic, coverage < classic, angle covering ρ ≈ 0. New read-out
  finding: the level scale refitted on the evaluated set makes labels depend on the other
  evaluated codes (seed 18 drop: 39 % vs 94 %); `iris.model.calibrated_accuracy` fits it on
  the training codes. Both reported.
* ECG full 7700-sample trace run: same shape as the 1800 window; Table I 2/3-neuron discrepancy
  stands.
* Codex review 005 of this session's diff: brief `notes/briefs/codex-05-session5-review.md`;
  status in the worklog entry (launched at close if not filed in `reviews/005-*.md`).

## Work queue, in order
1. **Codex review 005**: file the review as `reviews/005-session5.md`, fix or decline each finding
   with a resolution table, re-run `sh scripts/check-all.sh`, rebuild both PDFs.
2. **Send to the co-author** (Tobias): copy the `figures-needed.txt` figures next to the overleaf
   files, `latexmk -pdf standalone.tex`, fill [SLOTS] in the reply draft, send.
3. **ECG 2/3-neuron gap** (optional, workstation, ≈1.5–2.5 h — split it): the prototype-3 cell
   scores below prototype-2 on both windows, which points at optimisation (40 sweeps × 8
   restarts) rather than the model. Test with `--iterations 100` (the library default) on the
   1800 window before blaming the paper.
4. **E1 control** (optional): full-batch sample-wise training with ordinary sweeps
   (`--iterative-config` switch in `bitcount_figs.py`).
5. Shot-noise programme items 4–6.
6. Racetrack / Adam / multi-layer packages from the co-author: do not port; offer a review.

## Rules that bit us
* `uv sync --group dev` alone UNINSTALLS the optional extras (qiskit, cupy): use
  `uv sync --group dev --all-extras`. GPU tests need `LD_LIBRARY_PATH=/usr/lib/wsl/lib`.
* A pool script must set `OMP/OPENBLAS/MKL_NUM_THREADS` before `import numpy`, not in `main()`
  (48 workers × 64 BLAS threads = load 180). `tests/test_ecg.py` pins it for `ecg_table1.py`.
* Never run `pytest` without `-m "not slow"` here unless you mean it: the slow ECG tests take
  every core for minutes.
* Launch long runs from a detached `git worktree` at the commit (clean hash in meta.json while
  the main tree is being edited); symlink its `experiments/runs` to the main one. Remove the
  worktrees afterwards (`git worktree list`).
* The co-author's mail (attachments: ECG .mat, Overleaf zip, racetrack) is in Thunderbird on the
  Windows side; see `notes/inbox-2026-09-18.md` for what arrived.
* Grade a scheme in the observable it optimises; calibrate a read-out on training data.
