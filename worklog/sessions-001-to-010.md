# Session 1 — 2026-08-20 — kickoff: paper extraction, surveys, plan, scaffold

## HANDOFF — state at session close
(in progress)

## What was done
- Read PTQNN paper text and every figure (figure images extracted with `pdfimages`; bit-count
  target read off Fig 5 as (popcount+1)/112; Flappy = 7 qubits; ECG = 5 RX inputs + 1–3 prototypes).
- Located and downloaded prerequisite paper arXiv:2602.22241 (SNN) for KW/SA and Iris encoding.
- Surveys (opus subagents): orkan (C17, liborkan.so built, DM + 1q Kraus, no CRX/sampling, ≤14 q),
  Sturm.jl (channels-first IR, Choi-level tests, worklog sharding), QuantumHardware.jl (TOML
  devices, thin corpus, Quantinuum richest).
- Wrote PLAN.md, CLAUDE.md (=AGENTS.md), STATE.md, HANDOFF.md, this worklog; uv project.

## Verification
- `uv sync --group dev` OK.

## Gotchas for future agents
- Paper text says "6 hidden neurons" for Fig 4 but Fig 5's circuit has 5 hidden; both are bit-count.
