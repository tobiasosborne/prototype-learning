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
