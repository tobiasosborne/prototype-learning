# Session 1 — 2026-08-20 — kickoff: paper extraction, surveys, plan, scaffold

## HANDOFF — state at session close
(in progress — see STATE.md for the phase table)

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
- Still running at time of writing: Flappy, ECG, Iris, orkan Backend adapter, hardware layer,
  bit-count noise study.

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
