# Review 004 — session 4 (inbox round), Codex gpt-5.6-sol xhigh, read-only

Brief: `notes/briefs/codex-04-session4-review.md`; events `experiments/runs/codex/004-events.jsonl`; reviewed diff `afcfa48..a21b734`.

## Resolution table

(filled in below the review)

## Review (verbatim)

## BLOCKERS

1. **The iterative “three evaluations per iteration” claim is false for `train_iterative`.** Each outer iteration calls `kwsa.minimise` afresh: it evaluates the incumbent once at [kwsa.py:145](src/ptqnn/train/kwsa.py:145), then the two probes and candidate at [kwsa.py:165](src/ptqnn/train/kwsa.py:165). Thus one iteration costs four objective evaluations. The test itself acknowledges this at [test_bitcount.py:175](tests/test_bitcount.py:175).

   ```text
   train_iterative(..., iterations=1).evaluations == 4
   ```

   Therefore the “three circuit evaluations”, “109 versus 3”, and factor-36 claims at [04-reproduction.tex:57](reports/tex/sections/04-reproduction.tex:57), [A-assumptions.tex:14](reports/tex/sections/A-assumptions.tex:14), [app-agentic.tex:148](reports/tex/overleaf/app-agentic.tex:148), and [report_figures.py:265](experiments/report_figures.py:265) are wrong. Counting complete objective calls gives 109 versus 4, a factor 27.25.

2. **The bit-count artifacts mislabel minima as final losses, and one report range is false.** `res_i.loss` is the best monitored value, stored as `iterative_loss` at [bitcount_figs.py:147](experiments/bitcount_figs.py:147), then written under `final_kl` at [bitcount_figs.py:366](experiments/bitcount_figs.py:366). Examples:

   ```text
   seed 0: actual final 0.028142; reported “final” 0.017322
   seed 1: actual final 0.008901; reported “final” 0.000947
   ```

   See [histories.json:198](experiments/runs/bitcount/20260918T111515Z/histories.json:198) versus [meta.json:65](experiments/runs/bitcount/20260918T111515Z/meta.json:65). The prose’s end values are correctly derived from histories, but `meta.json` and `results.md` contradict them.

   Separately, “superposition KL … stays between 0.07 and 0.4 throughout” at [04-reproduction.tex:67](reports/tex/sections/04-reproduction.tex:67) and [app-agentic.tex:157](reports/tex/overleaf/app-agentic.tex:157) is false: stored values range from `0.035539` to `0.582940` ([histories.json:569](experiments/runs/bitcount/20260918T111515Z/histories.json:569), [histories.json:739](experiments/runs/bitcount/20260918T111515Z/histories.json:739)). Only the four final endpoints round to approximately 0.07–0.4.

3. **The ECG timing/protocol provenance does not support the report.** The run records `wall_seconds = -6179.47` at [meta.json:35](experiments/runs/ecg/20260918T132924/meta.json:35), while [04-reproduction.tex:216](reports/tex/sections/04-reproduction.tex:216) and [app-agentic.tex:276](reports/tex/overleaf/app-agentic.tex:276) claim 103 minutes on six cores. The metadata records neither process count nor host. The bug is use of non-monotonic `time.time()` at [ecg_table1.py:52](experiments/ecg_table1.py:52) and [ecg_table1.py:169](experiments/ecg_table1.py:169); several restart durations are negative too. One-line mechanical fix: replace all four elapsed-time `time.time()` calls with `time.perf_counter()`.

   The appendix also says ECG uses 100 sweeps at [A-assumptions.tex:11](reports/tex/sections/A-assumptions.tex:11) and [app-agentic.tex:777](reports/tex/overleaf/app-agentic.tex:777), but this run used 40 ([meta.json:15](experiments/runs/ecg/20260918T132924/meta.json:15)). The PSNR table itself is supported: 33.97/36.33/38.02 and 36.80/37.48/36.86 match [results.json:32](experiments/runs/ecg/20260918T132924/results.json:32), and input PSNR is 26.978 dB.

## MAJOR

1. **Dead per-basis models receive a mild, inconsistent KL penalty.** `activation_distribution` returns uniform when every activation is zero at [bitcount.py:125](src/ptqnn/tasks/bitcount.py:125), whereas the superposition route returns zeros specifically so the epsilon floor gives a large penalty. For zero parameters:

   ```text
   perbasis KL = 0.065406
   superposition KL = 26.310133
   ```

   This contradicts “same KL convention and eps handling” at [bitcount.py:136](src/ptqnn/tasks/bitcount.py:136). Returning zeros would preserve the existing dead-model convention.

2. **The documented monitor indexing fails at the final off-cadence point.** `train_iterative` explicitly permits a final record off cadence at [bitcount.py:209](src/ptqnn/tasks/bitcount.py:209), but the plot assigns every point `index × monitor_every` at [bitcount_figs.py:193](experiments/bitcount_figs.py:193). For five iterations monitored every two:

   ```text
   true x:    [0, 2, 4, 5]
   plotted x: [0, 2, 4, 6]
   ```

   The archived 100,000/1,000 run happens to be divisible and is unaffected; the API and general plotter are not consistent.

3. **The Iris drop policy leaks test labels into evaluation and inflates the reported accuracy.** [data.py:254](src/ptqnn/tasks/iris/data.py:254) independently identifies ambiguous codes using labels in each split, then removes test samples based on their test labels. For seed 0 it removes 18 of 100 test samples while reporting only three distinct dropped codes; both saved models score 97.56% on the filtered 82 samples but 95.0% on the original 100. This is a conditional evaluation, not an ordinary held-out accuracy. The assumption is not listed in `A-assumptions.tex` or the Overleaf assumptions, which still describe merging.

   `train_index`/`test_index` also remain unfiltered: lengths are 50/100 while the corresponding code arrays are 44/82 at [data.py:259](src/ptqnn/tasks/iris/data.py:259). No current drop-mode consumer indexes through them, but the `IrisSplit` invariant is broken.

4. **The claimed exact firing factorisation is tested only on marginals.** The commutation argument at [train.py:472](src/ptqnn/tasks/iris/train.py:472) is sound, including the un-computes, and the random prototype case does create entanglement. However [test_iris.py:393](tests/test_iris.py:393) compares only `P(hidden_j=1)` values. Checking every one-qubit marginal still cannot detect wrong hidden coherences or correlations—e.g. a Bell state and its dephased mixture have identical one-qubit marginals. This violates the repository’s explicit full-density/Choi testing rule and does not establish the claimed exact channel rewriting.

5. **The default relative dead-neuron test fails on an actually dead layer.** At [train.py:297](src/ptqnn/tasks/iris/train.py:297), all-zero drive gives `strongest=0`, so strict `drive < 0` marks nothing dead:

   ```text
   dead_hidden_neurons(net, zeros) == [False, False, False, False]
   ```

   Consequently `prototype_placement(...).all_dead` cannot detect an exactly decoupled layer under default tolerances.

6. **The firing null violates the project’s statistical discipline.** `firing_placement` defaults to only 200 draws at [train.py:619](src/ptqnn/tasks/iris/train.py:619), the archived rescore uses 200, and the statistical test uses 50 at [test_iris.py:483](tests/test_iris.py:483). The repository rule requires at least 1,000 samples and a stated tolerance. The reported z-scores therefore do not meet the project’s own evidentiary standard.

7. **`--rescore` cannot validate the provenance it publishes.** It requires the caller to reproduce seed/split/policy manually and does not read or verify an original `meta.json` at [iris_fig7_fig8_fig9.py:248](experiments/iris_fig7_fig8_fig9.py:248). The archived run has no training metadata; its `git_hash` is the rescoring commit, and the displayed configuration is reconstructed from current defaults. This directly contradicts “every run directory records which machine/code produced it” at [01-introduction.tex:42](reports/tex/sections/01-introduction.tex:42) and [app-agentic.tex:21](reports/tex/overleaf/app-agentic.tex:21). None of the three September metadata files records a machine; the Iris rescore records no wall time at all.

8. **Several diagnostic numbers have no reproducible run artifact.** The per-input KL `0.05`, input-register retention `1.3%`, and `0.03–0.09` comparison at [04-reproduction.tex:36](reports/tex/sections/04-reproduction.tex:36) are not tied to a run, and the archived iterative run does not save iterative parameters from which to recompute them. The same unsupported block appears at [app-agentic.tex:132](reports/tex/overleaf/app-agentic.tex:132).

9. **The main-report Fig. 6 citation is wrong.** [04-reproduction.tex:83](reports/tex/sections/04-reproduction.tex:83) says both panels use four seeds from the September run. The actual report generator loads the August six-seed run for the left panel at [report_figures.py:204](experiments/report_figures.py:204) and the September four-seed run only for the right. The Overleaf caption gets this right.

## MINOR

1. [kwsa.py:66](src/ptqnn/train/kwsa.py:66) documents the sole candidate as `x + a_n ĝ`; the minimisation update is `x − a_n ĝ`.

2. `n_coords`, `anneal_every`, and `batch` lack range validation. Negative `n_coords` silently means “all but some coordinates”; `anneal_every=0` divides by zero; direct `batch=0` constructs an empty mean and returns NaN at [bitcount.py:235](src/ptqnn/tasks/bitcount.py:235).

3. “Once per full pass over the basis inputs” at [bitcount.py:212](src/ptqnn/tasks/bitcount.py:212) is misleading: inputs are sampled with replacement, so `n // 2**n_inputs` is a nominal epoch, not a completed pass.

## VERDICT

Reject pending correction of the run accounting and report provenance. The central observable physics is correct: for input `|0⟩`, writing \(p=\sin^2\phi\,\sin^2(\theta/4)\) and \(r=\sin\phi\cos\phi\,\sin^2(\theta/4)\), the reduced input state after `RX(φ)–CRX(θ)–RX(−φ)` is \(\begin{pmatrix}1-p&-ir\\ir&p\end{pmatrix}\); for `|1⟩` the diagonal entries swap and the off-diagonal sign reverses, so the input is not generally restored. `basis_prep`, target-first KL, the nonzero activation distribution, `probe_candidates=False`, schedule keying, and the legacy default KWSA path check out; the ECG loader preserves the mailed arrays, measures sigma, handles `None`, and records correct digests. Focused non-slow bit-count/train, ECG, and selected Iris tests passed. No repository files were modified.

## Resolution table (session 4, same day)

| # | Finding | Resolution |
|---|---|---|
| B1 | "three evaluations per iteration" is four (incumbent re-scored per outer call) | **Fixed** in 04-reproduction, A-assumptions, app-agentic ("four objective evaluations", factor 27); figure footer via pi-06 |
| B2 | `final_kl`/results.md report the minimum; "0.07–0.4 throughout" false | **Fixed**: text now states range 0.035–0.58 with endpoints 0.07–0.4; `bitcount_figs.py` records `iterative_final` and `best` separately (pi-06). Archived `meta.json` of 20260918T111515Z keeps the old field; worklog gotcha notes it |
| B3 | negative `wall_seconds`; "103 min on six cores" unsupported; "ECG 100 sweeps" vs run's 40 | **Fixed**: `perf_counter` + `processes`/`hostname`/`platform` in meta (pi-06); prose drops the minutes, says "six worker processes on the laptop"; assumptions say 40 sweeps per restart, 8 restarts (August run's count to be confirmed on the workstation — HANDOFF) |
| M1 | dead per-basis model gets uniform instead of the epsilon penalty | **Fixed** (zeros; pi-06) |
| M2 | off-cadence final monitor point mis-plotted | **Fixed** (`monitor_x`, last point at `iterations`; pi-06) |
| M3 | drop policy filters the test set by its own labels (97.6 % on 82 vs 95.0 % on 100) | **Partly fixed**: both accuracies now reported; the policy itself is the co-author's ("removed binary patterns assigned to two classes") and stays as an `[ASSUMPTION]` with the unfiltered number beside it; A-assumptions entry added (below) |
| M4 | firing factorisation tested only on marginals | **Fixed**: full hidden-register density matrix vs reference partial trace, prototype and classic (pi-06) |
| M5 | all-zero drive → nothing dead | **Fixed** (pi-06) |
| M6 | 200 null draws < the 1000-sample rule | **Fixed**: default 1000, test ≥ 1000; the −drop run rescored with 1000 draws |
| M7 | `--rescore` provenance | **Fixed**: reads/validates `meta.json` when present, records origin of each value, hostname, wall time (pi-06). The 20260918T085641Z-drop run has no training meta (crashed before the fix) — stated in its `meta-rescore.json` |
| M8 | unarchived diagnostic numbers (0.05, 1.3 %, 0.03–0.09) | **Fixed**: removed from both texts; replaced by the archived dashed series; params now saved per seed for future diagnostics (pi-06) |
| M9 | Fig 6 caption vs generator | **Fixed**: generator reads the September run for both panels until the workstation regenerates (TODO in code + HANDOFF) |
| m1–m3 | docstring sign, range validation, "full pass" wording | **Fixed** (pi-06) |
