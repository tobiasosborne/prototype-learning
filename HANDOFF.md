# HANDOFF

Session 4 (2026-09-18) paused mid-flight on the laptop checkout. Read `STATE.md`, then
`notes/inbox-2026-09-18.md` (the co-author's requests and the Overleaf findings), then the
session-4 worklog entry. Deliverable now: the co-author's asks for the paper, in this order.

## First 15 minutes of the next session
`git status -s` shows uncommitted edits from three pi runs (briefs `notes/briefs/pi-01..03`).
For each: read the diff against its brief, run `uv run pytest tests/test_ecg.py` /
`tests/test_iris.py` / `tests/test_train.py tests/test_bitcount.py`, then
`sh scripts/check-all.sh`; commit per brief, or `git checkout --` the files and rerun the brief
(`docs/pi.md`). One known environmental failure: the hardware-plan state-prep test reads a run
dir that lives only on the workstation — skip it here, do not "fix" the number.

## Work queue, in order
1. **Real ECG Table I** (`pi-01`, minimum ask): once merged,
   `uv run python experiments/ecg_table1.py --source mailed --n-samples 1800 --processes 10`
   and again with `--n-samples 0` (full 7700). Report both rows against Table I
   (32.9/40.00/40.04). Input PSNR of the mailed data is ≈27.5 dB; do not retune `DEFAULT_SIGMA`.
2. **Iterative baseline** (`pi-03`): E0 re-score the old runs with the per-basis monitor;
   E1 `--iterative-batch 32 --iterative-iterations 20` (6 seeds, ~10 min); E2 single-coordinate
   `--iterative-iterations 100000 --iterative-monitor perbasis` (4 seeds, ~45 min on 10 cores).
   Then rewrite `reports/tex/sections/04-reproduction.tex` "iterative baseline" paragraph and the
   scoreboard rows in `reports/tex/overleaf/`.
3. **Iris drop policy** (`pi-02`): `experiments/iris_fig7_fig8_fig9.py --ambiguous drop`, all
   three splits; compare accuracy/ceilings to the merge policy.
4. **Fig 9 metric**: implement the normalised 4-medoid covering cost ρ (design in the session-4
   Opus analysis, quoted in the worklog Gotchas is only the physics; the metric spec is in the
   agent report — re-derive: decode each hidden neuron to a 12-bit code by per-feature argmax of
   cos²(φ/2), R(B) = mean over distinct training codes of min_j d_H/2, null = random 4-subsets of
   the 81 patterns, floor = best 4-subset of the codes, ρ = (R_null−R)/(R_null−R*)). pi brief.
5. **Codex adversarial review** (gpt-5.6-sol, xhigh) of `reports/tex/overleaf/` + items 1–3,
   filed as `reviews/004-inbox.md`. Then copy the figures from `figures-needed.txt` and hand
   the folder to the co-author.
6. **Reply to the co-author**: figure questions (10 right, 11, 12, 14 right, 16, 17/18), shots
   (report + SPRT, no further optimisation), costs (simulators only, nothing paid), the
   iteration-unit finding, and the two decisions only Tobias can take (authorship line; making
   the GitHub repo public for the `\repourl` link).
7. Shot-noise programme items 4–6 (session-3 queue) stay below all of the above.
8. Racetrack / Adam / multi-layer packages: do not port; offer a review.

## Rules that bit us
* Laptop: no GPU, no orkan, 2 P-cores + 8 E-cores; budget runs at ≤ 5 core-equivalents and
  never run three pi agents' test suites concurrently with a timed pytest.
* pi in `-p` mode is silent until exit; two pi runs must have disjoint file lists.
* Everything from the co-author's inbox is local-only (scratchpad / `data/raw`), not committed.
