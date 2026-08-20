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
