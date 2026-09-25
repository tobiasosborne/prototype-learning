# Session 5 — 2026-09-25 — workstation round: review-004 fixes, laptop runs regenerated, Iris 20-seed sweep, ECG full trace

## HANDOFF — state at session close
Workstation (64 cores, RTX 3090, orkan). Review 004 closed; every laptop run regenerated here from
clean commits and bit-identical where it overlaps; HANDOFF queue items 1–3 of session 4 done. Fig 9
scored over 20 seeds × 2 policies against a pre-registered criterion: not reproduced. New Iris
read-out finding (set-refitted scale). Main report and Overleaf updated; Codex review 005 pending.

## What was done
- Mailed ECG `.mat` files recovered from Thunderbird (Windows profile, uni-hannover INBOX,
  Rosenhahn 2026-08-21) into `data/raw/ecg-rosenhahn/`; SHA256 match the pinned digests.
- Review-004 code fixes by two Opus subagents on disjoint files (bit-count/KW–SA/ECG/figures;
  Iris), each red→green; re-tested and committed here (`a2dd8da`, `4915fa2`). Found en route:
  `ecg_table1.py` set `OMP_NUM_THREADS` after numpy import (48 × 64 BLAS threads, load 180) →
  `6c8857f` with a pinning test. Stale "three circuit evaluations" in two captions fixed.
- Runs (all from detached worktrees at the named commit): `bitcount/20260925T153548Z` (E2, 6 seeds,
  44 min; seeds 0–3 = laptop bit for bit; superposition series = August run bit for bit),
  `ecg/20260925T153739Z-mailed1800` (77 min on 48 procs; = laptop in every cell),
  `ecg/20260925T165741Z-mailedfull` (39 min on 24 procs), Iris sweep `iris/20260925T*-seed*-*`
  (40 runs, 20 at a time × 2 procs, ≈30 min each).
- `experiments/iris_sweep_summary.py` (+ tests): pre-registered verdict rules, both accuracies,
  Fig 7 contrast; `--calibrated` adds `iris.model.calibrated_accuracy` (scale fitted on the
  training codes). Summary `iris_sweep/20260925T173530Z`.
- Report: Fig 6 regenerated (both panels, one six-seed run); §4 bit-count, Iris ("Fig. 9,
  scored", "Ambiguous codes, and how labels are read out"), ECG (full-trace row); scoreboard,
  learnings (new item 9), assumptions, provenance, compute paragraph. Overleaf mirrored by an Opus
  subagent. STATE, reply draft, review-004 resolution table, experiments/README pinned runs.

## Verification
- `sh scripts/check-all.sh` (see close entry below for the final run). Main report builds, 32 pp,
  0 undefined references, no new overfull boxes.
- Every new number re-derived from run files in this session: E2 crossings (12k/23k/62k/73k),
  ECG tables (stdout.log/results.json), sweep summary.json. Seed-18 anomaly traced: set-refitted
  and per-sample set-refitted read-outs agree on all 80 model×run cells; only the scale-fit set
  matters.

## Gotchas for future agents
- `uv sync --group dev` without `--all-extras` uninstalls qiskit/cupy (20 tests then skip).
- `pytest` without `-m "not slow"` runs the slow ECG tests on every core.
- Thread caps must precede `import numpy` in pool scripts.
- The set-refitted Iris accuracy (`accuracy`, `test` in results.json) is transductive; quote
  `calibrated_accuracy` as the held-out number.
- Prototype-3 ECG scores below prototype-2 on both windows: suspect optimisation budget first.

# (earlier: session 4)

# Session 4 — 2026-09-18 — inbox round: co-author's asks, real ECG, iterative baseline recast, Iris policy + placement metric, Overleaf drop-in

## HANDOFF — state at session close
Laptop checkout (12 threads, no GPU, no orkan). The co-author's minimum asks are done and
written up: ECG on his original recording, the iterative baseline in its own observable, his
Iris "drop" policy, a quantitative replacement for the Fig 9 t-SNE, and a compiling Overleaf
drop-in with the reproducibility subsection and appendix. Codex review 004 in progress at
close (see `reviews/004-inbox.md` if present). Queue: `HANDOFF.md`.

## What was done
- Inbox triage (3 Sonnet summarisers) → `notes/inbox-2026-09-18.md`; .mat files staged under
  `data/raw/ecg-rosenhahn/` (SHA256 in `tasks/ecg/data.py`).
- pi-01 (deepseek-flash): `--source mailed` ECG loader, sigma measured, provenance in meta.json.
  pi-02 (glm-5.3-flash): Iris `ambiguous="drop"` (`[Rosenhahn e-mail 2026-08-21]`).
  pi-03 (deepseek-flash): `KWSAConfig.n_coords/probe_candidates`, `activation_distribution`,
  `perbasis_kl_loss`, `ITERATIVE_CONFIG`, `train_iterative(monitor=, batch=, monitor_every=)`.
  pi-04 (glm-5.3-flash): covering-cost placement metric, `--ambiguous` flag.
  pi-05 (deepseek-flash) hung for 85 min without output → killed; figure written by hand.
- Opus: (a) iterative-baseline diagnosis (observable mismatch + probe acceptance + per-example
  schedule; measured per-eval costs); (b) Overleaf drop-in `reports/tex/overleaf/`; (c) report
  text passes (iterative, ECG, compute provenance); (d) firing-based Iris placement score with
  an exact per-input-qubit Kraus factorisation (13 ms vs 2.5 s per parameter vector), relative
  dead-neuron tolerance, crash-proof scoring and `--rescore`; (e) reply draft
  `notes/reply-to-bodo-draft.md` (all numbers filled, [SLOTS] left).
- Runs (all laptop): `bitcount/20260918T111515Z` (E2: 4 seeds × 1e5 single-coordinate steps,
  83 min), `ecg/20260918T132924` (mailed, 1800 samples, 103 min), `iris/20260918T085641Z-drop`
  (seed 0; crashed in scoring after training, rescored from params), `bitcount/…-fullbatch`
  (E1 with the single-coordinate config: inconclusive, KL 0.06→0.04 in 720 steps; not reported).
- Report: 04 iterative paragraphs rewritten, `tab:ecg-mailed`, compute-discipline paragraph now
  names both machines, B-provenance lists the .mat digests, scoreboard rows updated, Fig 6
  right panel regenerated (`RUNS["bitcount_iterative"]`). Overleaf files mirror all of it.
- Fixes en route: `bitcount_figs.py` x-axis (index × monitor_every), Iris scoring order
  (results before scoring), two hardware-plan tests skip without the workstation run dirs,
  `docs/codex.md` home path, Codex CLI reinstalled, `docs/pi.md` written.

## Verification
- `sh scripts/check-all.sh`: ruff/format/mypy clean; pytest 457 passed, 15 skipped, 6 deselected
  (before the skip marker: 2 failed, both reading `experiments/runs/iris/20260820T*` which exist
  only on the workstation).
- `reports/tex/main.tex` → 30 pp, 0 errors, 0 undefined refs; `overleaf/standalone.tex` → 16 pp,
  0 errors (one pre-existing overfull hbox in the fidelity display).
- Every number in the report was re-derived by the writing agent from the run files and
  spot-checked here (E2 crossings 12k/23k/73k; ECG 33.97/36.33/38.02, 36.80/37.48/36.86).
- Codex review 004 filed (`reviews/004-inbox.md`): 3 blockers, 9 majors; physics of the observable
  mismatch confirmed by an independent 2-qubit derivation. LaTeX findings fixed the same day;
  code findings open (pi-06 hung 3 h with no edits, killed). ≥20-seed Iris sweep not run.

## Gotchas for future agents
- The prototype block RX(φ)·CRX·RX(−φ) does NOT restore the input register, so the Eq. 4
  superposition KL and the per-input activation KL are different observables (0.03–0.4 apart on
  the same network). Grade a scheme in the observable it optimises.
- `kwsa.minimise` accepts the ±c probes as candidates by default: fine for batch objectives, a
  random walk on a one-example objective (`probe_candidates=False`).
- Old `train_iterative` cost 613 ms/iteration (109 evals); with `n_coords=1` 16.7 ms.
  `meta.json["final_kl"]["iterative"]` is the MINIMUM of the history, not the last value.
- Iris output couplings are small (amplitude-encoded target ∝ (label+1)²): any absolute
  "dead neuron" threshold kills every neuron; use the relative one.
- Trained Iris prototypes are anti-patterns (angles ≈ ±π on 2–4 inputs, ≈0 elsewhere): one-hot
  argmax decoding is ill-posed for them; the firing-based score is encoding-free.
- Mailed ECG: noisy trace clipped to [0,1], noise std 0.035 (not our 0.02), input PSNR 27.0 dB.
  `ecg_table1.py` stores `wall_seconds` negative (clock bug; per-cell seconds are right).
- pi (`-p`) prints nothing until exit; `pkill -f '<pattern>'` from inside a tool shell whose own
  command line contains the pattern kills that shell. Kill by PID from `pgrep -x pi`.
- Orphaned pytest children of killed agents kept 6 cores busy for an hour: check `ps` before
  reading a wall time as a per-evaluation cost.

# (earlier: session 3)

# Session 3 — 2026-08-21 — documentation round

## HANDOFF — state at session close
No code or report numbers changed. The repo now has layer docs (`docs/`) and a README in every
folder; `CLAUDE.md`/`AGENTS.md` point at `docs/README.md`. Work queue unchanged: shot-noise
programme items 4–6 (HANDOFF.md).

## What was done
- `README.md` (was empty; `pyproject` readme target): install, a runnable 30-second tour
  (checked: clean KL 5.5e-4, ibm_fez 3.6e-3), layout, reproduction commands, conventions.
- `docs/README.md` index; `docs/architecture.md` (package map, three invariants, network
  parameter layout, data flow, fast paths, where assumptions live); `docs/backends.md` (backend
  matrix with caps and what each is pinned to; orkan 2q-channel routes; CRN); `docs/noise.md`
  (channel algebra, NoiseModel placement/ASAP semantics, `scaled`, preset mapping, the seven
  devices and their gaps, provenance rules); `docs/training.md` (KW–SA config table, per-task
  schedules, the superposition observable, finite-shot training: variance/bias/decision noise);
  `docs/tasks.md` (per task: paper fixes vs our assumptions, module map, results, scripts);
  `docs/hardware.md` (three transpile passes, routing caveat, QASM bit order, executors, cost
  model, `hardware_plan.py` sections); `docs/testing.md` (rules, per-file map, how to add).
- Folder READMEs: `experiments/` (script table with costs, pinned run-dir table with commits,
  rules), `reports/` (what each report is, how to build the PDF, latexmk gotchas), `reviews/`
  (round table with headline findings and status, how to write a brief), `notes/`, `tests/`,
  `data/` (MIT-BIH 100 provenance/licence), `literature/` (what is gitignored and why).
- `CLAUDE.md` = `AGENTS.md`: one line pointing at the docs (74 lines).

## Verification
- `LD_LIBRARY_PATH=/usr/lib/wsl/lib sh scripts/check-all.sh`: green — 441 passed, 6 deselected
  (34.8 s). README snippet executed end to end.
- Every function/flag/number named in the new docs was read from the source or `STATE.md`
  during writing; run-dir commits from the `meta.json` files.

## Gotchas for future agents
- `docs/*.md` describe layers and deliberately repeat no moving numbers beyond what `STATE.md`
  holds; when a number changes, change `STATE.md` and the report, and grep `docs/` for it.
- `tests/oracle/indep_sim.py` must stay free of `ptqnn` imports (blind-oracle property; stated
  in `tests/README.md`).

# (earlier: session 2)

# Session 2 — 2026-08-21 — review-002 queue, shot-noise programme, provenanced hardware plan, LaTeX report

## HANDOFF — state at session close
Deliverable: `reports/tex/main.pdf` (28 pp, 23 figures regenerated by `experiments/report_figures.py`).
Review-002 queue closed (B1 hardware numbers, #7 shared batched kernel, #8 restate_incumbent, #9 Iris
wording); hardware plan v2 is script-generated with per-field provenance; shot-noise items 1–3 done
(margin training negative, CRN partial, SPRT positive: ÷4.6 shots at 100 % perfect play); Codex review
round 3 launched over the session diff — findings in `reviews/003-session2.md`, resolution status in
"Verification" below. Work queue = HANDOFF.md. No runs longer than 11 min were executed.

## What was done
- Fixed the GPU density-matrix regression (`superoperator` forced a host conversion under CuPy 14).
- `experiments/hardware_plan.py`: every number of `reports/hardware-plan.md` from one run dir
  (`runs/hardware-plan/20260821T084554Z`): compiled counts per real preset, fidelity budgets,
  simulated TVD of *compiled* circuits (5–15× the logical ones), qiskit O1–O3, costs, shot budgets,
  state-prep counts, and a provenance join that refuses to run on an unsourced field; assumptions
  A1–A9 enumerated. Report rewritten; v1 errors listed in reviews/002 resolution.
- `backends/batched.py`: one batched statevector kernel (any 1q/2q gate) replaces the three private
  rx/crx kernels in ECG/Flappy/Iris; pinned to the reference on full distributions.
- Flappy `train(batch>0)` re-scores the incumbent and returns the final point: rotating batch gives
  25/25 perfect (was 0/25 — stale-incumbent artefact). Iris split wording corrected.
- Shot-noise programme: `ShotBackend` with maximal-coupling common random numbers (shared
  inverse-CDF uniforms gave only 1.3×); `kwsa.minimise(probe_context=…)`; `SPRTDecision`; per-frame
  margin penalty; policy records frame margins and supports sampled decisions.
  Experiments: `bitcount_shots.py` (126 s), `flappy_shots.py` (647 s). Results in reports/noise.md
  "Shots" and STATE.md.
- reports/noise.md: ECG/Iris reduced run folded in; 11-q noise-aware grid marked not run (13 h).
- LaTeX report `reports/tex/` (no jargon, notation section, assumptions + provenance appendices).
- Codex (gpt-5.6-sol, read-only) adversarial review of `7271620..HEAD` (brief
  notes/briefs/codex-03-session2-review.md).

## Verification
- `LD_LIBRARY_PATH=/usr/lib/wsl/lib sh scripts/check-all.sh`: green at 1bc13d2 and at session close
  (see final entry below). GPU tests green. 441 fast tests.
- Review 003 (Codex, read-only, 1 blocker / 4 major / 4 minor): all nine applied in commit
  `review-003 fixes` and the follow-up; resolution table at the end of reviews/003-session2.md.
  Blocker 1 changed a conclusion: Iris state preparation was budgeted for 120 samples instead of
  the 12–14 distinct codes actually prepared (9× over-count) — "Iris is out of reach" withdrawn.
  Major 2: the provenance validator now compares values with exact field ids (it immediately found
  two parser gaps the substring version had hidden). Major 5: idle relaxation charged over
  schedule gaps only (Fez Flappy F 0.40 → 0.46, bit-count 0.043 → 0.068; rankings unchanged).
  Both shot experiments re-run at a clean commit: tables identical, loss arrows now comparable.

## Gotchas for future agents
- `np.asarray` on a CuPy array raises under CuPy ≥ 13; keep `__array_function__` dispatch paths free
  of host conversions (`reference.superoperator`).
- A shaped-objective penalty that can exceed the per-frame reward selects for dying early
  (Flappy margin hinge, run flappy_shots/20260821T075048Z).
- SPRT indifference margin must sit below the margins that decide the game (p10 ≈ 0.005), not at the
  median (0.02); at the median it loses to fixed-N.
- Sampled KW–SA training is limited by the plug-in KL bias (K−1)/(2·P(out=1)·N), not by gradient
  noise once CRN is on.
- Background bash commands piped through `tail` show nothing until they exit; log to a file instead.
- latexmk: `\ket` is predefined by quantikz (use `\renewcommand`); `\path/\url` cannot sit in
  captions; a global caption `width=` breaks side-by-side minipage figures.

# (earlier: session 1)

# Session 1 — 2026-08-20 — kickoff: paper extraction, surveys, plan, scaffold

## HANDOFF — state at session close
All four paper experiments reproduced (bit-count Fig 3/5/6, Iris Fig 7/8, Flappy, ECG Table I/Fig 13)
with discrepancies recorded in reports/reproduction.md; noise library + 7 sourced vendor presets +
orkan/GPU/trajectory backends + hardware layer in place; bit-count and Flappy noise studies done,
ECG/Iris noise and 11-qubit noise-aware training not run (budget). Work queue = HANDOFF.md.
Reviews: 001 (applied), 002 (partially applied — blocker 1 = hardware-plan numbers OPEN),
002b Codex (applied). 393 tests; `sh scripts/check-all.sh` green except when the GPU is saturated.

## What was done
- Read PTQNN paper text and every figure (figure images extracted with `pdfimages`; bit-count
  target read off Fig 5 as (popcount+1)/112; Flappy = 7 qubits; ECG = 5 RX inputs + 1–3 prototypes).
- Located and downloaded prerequisite paper arXiv:2602.22241 (SNN) for KW/SA and Iris encoding.
- Surveys (opus subagents): orkan (C17, liborkan.so built, DM + 1q Kraus, no CRX/sampling, ≤14 q),
  Sturm.jl (channels-first IR, Choi-level tests, worklog sharding), QuantumHardware.jl (TOML
  devices, thin corpus, Quantinuum richest).
- Wrote PLAN.md, CLAUDE.md (=AGENTS.md), STATE.md, HANDOFF.md, this worklog; uv project.
- Core (main session, TDD): circuit IR (CP-map first, unitarity/TP witnesses, qubit 0 = MSB),
  numpy sv/dm reference backend, neurons + Network, KW–SA optimiser (coordinate-sweep mode
  found necessary: whole-vector KW stalls), bit-count task, KL/cosine/ℓ2 losses, noise channel
  library (Choi-level tests), NoiseModel with ASAP idle scheduling (Aer cross-check), presets
  loader, trajectory backend, CuPy GPU backends.
- Codex (gpt-5.6-sol): blind independent DM simulator → tests/oracle; reference agrees to 1e-12.
- Subagents (opus): orkan ctypes binding (12q CRX 20 ms, LSB-0 native, 32 threads optimal), paper
  distillations (docs/physics; found θ = 2·asin(√s) pin in SNN Fig 4), 7 sourced vendor device
  TOMLs, bit-count Fig 3/5/6 script + run, review 001 (8 majors fixed, see reviews/).
- Tasks (opus subagents, all TDD, reports in reports/reproduction.md):
  Flappy — headless clone, 7-qubit policy, perfect play 110/110 pipes on 20/20 held-out seeds,
  8/8 optimiser seeds; Iris — (label+1)² target found from Fig 8 bar heights, acc 91–98 %,
  Fig 7 overfitting contrast reproduced, Fig 9 not; ECG — Table I within ~2 dB, Fig 13
  reproduced, θ=π·s ablation −0.6…−7 dB.
- Backends: orkan DM adapter (36× numpy at 11 q), block-superoperator rewrite of the reference
  DM (numpy 148 s → 33 s, GPU 18 s → 1.5 s per noisy 11-q evaluation), CuPy DM + trajectory.
- Noise: channels (Choi-level tests), NoiseModel (ASAP idling; matches Aer delay-for-delay),
  presets with provenance, scaled(); Codex adversarial review: physics clean, 3 robustness fixes.
- Hardware: gate sets, CRX/1q/2q decompositions (verified mod phase), greedy router, QASM3,
  executors (local/Aer/dry-run), compile table (30 CRX → 60/327/540 native 2q), hardware plan.
- Noise studies: bit-count (prototype better under noise only via better clean fit; 2q budget
  dominates; KL ∝ s^1.8; shot-noise bias), Flappy (100 % perfect under all presets; shot noise
  needs ≥1e4 shots/frame), ECG/Iris partial (17-q trajectories 0.3–0.7 s each → 8 h full grid).
- Review 002 (opus): 2 blockers, 10 majors; GPU regression fixed, presets now refuse unpublished
  error fields, hardware fixes (#3,4,6,10,11) time-boxed; hardware-plan numeric errors OPEN.

## Verification
- `uv run pytest -m "not slow"`: green at each commit (see git log); slow suite incl. Aer
  cross-check and bit-count convergence green at 97b7ad0; GPU tests green with
  `LD_LIBRARY_PATH=/usr/lib/wsl/lib`; orkan ffi suite green.

## Gotchas for future agents
- Paper text says "6 hidden neurons" for Fig 4 but Fig 5's circuit has 5 hidden; both are bit-count.
- `codex exec` launched with a trailing `&` inside the Bash tool dies when the tool returns; use
  `run_in_background`.
- A `git add -A` while subagents are mid-flight commits their half-finished files; add paths
  explicitly.
- The iterative Fig-6 baseline is under-specified; our per-sample sweep does not reach KL≈0.
- `presets.device(name)` raises for rigetti_ankaa_3 unless `allow_missing=True`; the noise
  scripts that loop over `available_devices()` predate this and need the flag.
- GPU tests fail with CuPy OOM when another process holds the 24 GB pool; rerun alone.
- Noise-aware training cost at 11 q: 108 evals/sweep × 1.5–3 s → ≥ 40 min per cell. Budget it.
- `restate_incumbent` exists in kwsa but Flappy seed rotation is not wired to it yet.
