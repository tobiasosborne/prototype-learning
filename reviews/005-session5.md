# Review 005 — session 5 (workstation round), Codex gpt-5.6-sol xhigh, read-only

Brief: `notes/briefs/codex-05-session5-review.md`; events `experiments/runs/codex/005-events.jsonl`;
reviewed diff `5abedbe..afc13fe`.

## Resolution table (session 5, same day)

| # | Finding | Resolution |
|---|---|---|
| B1 | z-scores read as a between-architecture comparison; raw values rank the other way on selectivity | **Fixed.** Verified (raw medians, 20/20 and 19/20, 16/20 counts). `iris_sweep_summary.py` now reports raw coverage/selectivity, their untrained medians and per-seed counts (test). Summary regenerated: `iris_sweep/20260925T175903Z`. Text rewritten in 04 (Fig. 9 result), scoreboard, learnings, assumptions, Overleaf (app-agentic, sec-reproducibility), STATE, reply: the prototype layer covers more, but already does so untrained — the effect is the architecture's; classic is more selective; verdict unchanged (not reproduced) |
| B2 | "angle score zero for both" false (classic direct/merge CI excludes 0) | **Fixed.** Classic bootstrap CI added (test); text restricts the zero-containing claim to the prototype and names the classic interval |
| B3 | reply: "our 1e5 steps are your 120" arithmetic wrong | **Fixed.** Both readings stated: elementary step ⇒ ×1000 as suggested; sweep ⇒ 120 × 36 = 4,320 steps, where our seeds are at KL 0.02–0.05 |
| M1 | "every run directory records which machine" unsupported | **Fixed.** Scoped to the 25-September runs; earlier ones attributed by date; laptop overlap stated as agreement with the recorded values (laptop dirs not on the workstation) |
| M2 | Fig 7 "monotone" vs a 5 % endpoint rule | **Fixed.** Rule stated; merge/drop counts separated; prototype upticks on 9/20 stated |
| M3 | unmatched glob patterns silently ignored | **Fixed.** `resolve_dirs` raises on an unmatched pattern or none (test) |
| M4 | ECG full trace does not "rule out the window" | **Fixed.** "not peculiar to that window under this protocol" |
| M5 | read-out lesson conflates transductive with leakage; 13 flowers, not 3 | **Fixed.** Learnings item retitled "transductive", no test labels used, "13 flowers carrying three codes"; assumptions say "inductive" |
| m1 | `assert_allclose` default rtol | **Fixed.** `rtol=0` on both hidden-state pins |
| m2 | 36.25 → 36.2 (not 36.3) | **Fixed** in scoreboard, Overleaf, reply |
| m3 | windowed SSIM range 0.84–0.94 | **Fixed** in both tables/captions |

## Review (verbatim)

## BLOCKERS

1. **The Fig. 9 interpretation reverses the measured architecture ranking.** [04-reproduction.tex:137](reports/tex/sections/04-reproduction.tex:137) concludes that prototypes are more selective while classic neurons cover more data. Recomputing medians from the 40 `results.json` files gives:

   ```text
                    coverage P/C       selectivity P/C
   merge            0.894821/0.630594   0.147472/0.454702
   drop             0.804420/0.601012   0.141418/0.416469
   ```

   Classic selectivity exceeds prototype selectivity on **all 20 seeds under both policies**. Prototype coverage exceeds classic coverage on 19/20 merge and 16/20 drop seeds. The architecture-specific null means and standard deviations differ; comparing their z-scores measures standardised training improvement, not absolute coverage or selectivity. The predefined verdict remains negative, but its physical interpretation is wrong throughout the scoreboards, learnings, STATE and reply draft.

2. **“Angle score zero within error for both networks” is false.** [04-reproduction.tex:140](reports/tex/sections/04-reproduction.tex:140), duplicated at [app-agentic.tex:231](reports/tex/overleaf/app-agentic.tex:231). The aggregator bootstraps **only prototype medians** at [iris_sweep_summary.py:152](experiments/iris_sweep_summary.py:152). Applying the identical procedure—10,000 resamples, seed 0—to classic/direct scores gives:

   ```text
   merge, classic/direct:
   median rho = 0.1355477884
   bootstrap 95% CI = [0.07321595, 0.22084409]
   ```

   Zero is excluded. Restrict the statement about zero-containing intervals to prototypes; those intervals also establish uncertainty, not equivalence to random patterns.

3. **The reply’s iteration conversion is arithmetically wrong.** [reply-to-bodo-draft.md:23](notes/reply-to-bodo-draft.md:23) says a sweep over all weights makes “our 1e5 steps … your 120.” There are 36 coordinates:

   ```python
   100_000 / 36   # 2777.777777777778 sweeps
   120 * 36      # 4320 elementary steps
   ```

   The email’s “1000 times more iterations” does not establish the claimed sweep conversion.

## MAJOR

1. **The universal machine-provenance claim remains unsupported.** [01-introduction.tex:42](reports/tex/sections/01-introduction.tex:42) and the Overleaf introduction still say every run records its machine. August bit-count, Iris and ECG metadata contain no hostname/platform/worker count. All three cited September-18 directories are absent from this checkout, preventing independent verification of the claimed bit-identical laptop overlap. Scope the mechanical-provenance claim to the new runs and distinguish historical attribution from recorded evidence.

2. **The Fig. 7 counts do not establish monotonicity.** [01-introduction.tex:70](reports/tex/sections/01-introduction.tex:70) and [STATE.md:35](STATE.md:35) retain “prototype monotone.” The implemented rule is `final > 1.05 * minimum`, not “ever rises.” Prototype histories are non-monotone on **9/20 seeds under each policy**. For example:

   ```text
   drop, seed 0, sweeps 38→39: 0.0377520191 → 0.0388270527
   counted as “rises”: false
   ```

   State the 5% endpoint criterion explicitly. The quoted 8/2 counts concern merge; drop gives 7/0.

3. **Unmatched input patterns disappear silently.** [iris_sweep_summary.py:309](experiments/iris_sweep_summary.py:309). Supplying the valid merge pattern plus a misspelled `*-drpo` pattern exits successfully and writes a 20-run, merge-only summary without warning. Missing-file tests do not cover this omission. Reject unmatched requested patterns and empty selections.

4. **The ECG check does not “rule out the window.”** [04-reproduction.tex:279](reports/tex/sections/04-reproduction.tex:279), repeated in Overleaf, STATE and the reply. The comparison changes training/test blocks from 898/898 to 3848/3848 windows, the sampled training examples, and PSNR range from 0.789595 to 0.845038. It demonstrates persistence on these two extents under the chosen protocol; it does not exclude window/protocol effects generally.

5. **The new read-out lesson conflates transductive evaluation with label leakage.** [08-learnings.tex:62](reports/tex/sections/08-learnings.tex:62). `fit_level_scale` uses no test labels. Its demonstrated defect is batch-dependent prediction; describe the result as transductive rather than an ordinary inductive held-out accuracy. The separate drop filter does consult test labels. Also, the example removes **13 flowers representing three codes**, not three flowers.

## MINOR

- [test_iris.py:513](tests/test_iris.py:513): the advertised `1e-12` matrix pin retains NumPy’s default `rtol=1e-7`. A `1e-8` perturbation of an entry equal to `0.5` passes. Set `rtol=0` for the claimed absolute bound.
- [01-introduction.tex:74](reports/tex/sections/01-introduction.tex:74): full-trace prototype-3 PSNR is `36.2464545`, which rounds directly to **36.2**, not 36.3. The same double-rounding appears in the Overleaf scoreboard and reply.
- [04-reproduction.tex:304](reports/tex/sections/04-reproduction.tex:304): after adding the full-trace rows, windowed SSIM reaches `0.9359113`; the two-decimal range is **0.84–0.94**, not 0.84–0.93.

## VERDICT

Reject pending correction of the statistical interpretation and report claims. The 128 focused tests passed; checks against the parent reproduced the repaired failures. All 24 bit-count parameter files and 12 ECG table cells reproduce their recorded values; the regenerated Iris summary matches exactly. Wilcoxon direction and prototype bootstrap calculations match the stated rules. Seed 18 reproduces 39.08% filtered, 94% unfiltered and 96.55% filtered after training calibration. Its normaliser over distinct training codes is consistent, self-calibration equivalence holds, and calibration itself uses no test data. Both read-outs yield identical labels across all 80 saved models, despite differing probabilities. No repository files were changed.
