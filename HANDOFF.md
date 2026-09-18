# HANDOFF

End of session 4 (2026-09-18, laptop). Read `STATE.md`, then `notes/inbox-2026-09-18.md`
(what the co-author asked for and what his Overleaf draft contains), then the session-4 worklog
entry. Deliverables now: `reports/tex/overleaf/` (drop-in for the co-author's Overleaf),
`notes/reply-to-bodo-draft.md` (e-mail; fill [SLOTS]), `reports/tex/main.pdf` (30 pp).

## Open at close
* Codex review 004 is filed in `reviews/004-inbox.md` with a resolution table. The LaTeX
  findings are fixed and committed. The CODE findings (B2, B3-code, M1, M2, M4–M7, M9, m1–m3,
  and the "3 evaluations" figure footer) are OPEN: the pi-06 run on them hung for 3 h with no
  edits and was killed. Rerun `notes/briefs/pi-06-review-004-code.md` (split it in two if a
  single pi run hangs again: bitcount/kwsa/ecg items, then iris items), verify, then rescore
  `experiments/runs/iris/20260918T085641Z-drop` with 1000 null draws and refresh the Iris
  numbers in STATE.md and `notes/reply-to-bodo-draft.md`.
* Authorship: settled — acknowledgement only (Tobias, 2026-09-18). Do not reopen.
* GitHub visibility for the `\repourl` link: not checked (`gh repo view --json visibility`).

## Work queue, in order (workstation unless marked)
1. **ECG full trace**: `experiments/ecg_table1.py --source mailed --n-samples 0` (7700 samples;
   ≈4× the 103-min laptop run → ≈30 min on 48 procs). Add as a row to `tab:ecg-mailed`.
2. **Iris ≥20-seed sweep, both policies**: `experiments/iris_fig7_fig8_fig9.py --seed S
   --ambiguous {merge,drop}` for S in 0..19 (≈10 min each on the workstation, 2 procs; run
   10 at a time). Then the Fig 9 verdict per the criterion in `iris_train.firing_placement`
   (both z-scores positive for the prototype and above the classic; paired Wilcoxon) and per
   `prototype_placement` (median ρ ≥ 0.5). Write the result into 04-reproduction (Fig 9
   paragraph), the scoreboard rows, and the overleaf files. One seed says: prototype
   selectivity z +21 vs classic +0.2; coverage z +0.65 vs +3.0; ρ +0.03/+0.29 vs −0.17.
3. **Fig 6 left panel**: regenerate `fig_bitcount_convergence` on the workstation where
   `RUNS["bitcount"]` (6 seeds) exists; then restore "six seeds" in the caption
   (`04-reproduction.tex` l.84 and l.88 note).
4. **E1 control properly**: full-batch (32 inputs) sample-wise training with ordinary sweeps
   (SUPERPOSITION_CONFIG-like, probes on) — needs a `--iterative-config` switch in
   `bitcount_figs.py`; the Opus pilot reached per-input KL 1e-3 in 12 sweeps. Optional.
5. **Send to the co-author**: after 1–3, copy `figures-needed.txt` figures next to the overleaf
   files, re-run `latexmk` on `standalone.tex`, hand over the folder + e-mail.
6. Shot-noise programme items 4–6 (session-3 queue) stay below all of the above.
7. Racetrack / Adam / multi-layer packages from the co-author: do not port; offer a review.

## Rules that bit us this session
* Laptop budget: one experiment at a time; `OMP_NUM_THREADS=1`; check `ps` for orphaned
  pytest children before trusting any wall time. Three pi test suites + two experiments gave
  load 43 on 12 threads.
* Never `pkill -f '<pattern>'` with a pattern that is on the calling shell's own command line.
* pi `-p` is silent until exit; a pi run with no `git status` change after 30 min is hung.
* Write results before scoring; scoring code that raises must not be able to lose a run.
* Grade a training scheme in the observable it optimises (superposition KL ≠ per-input KL).
