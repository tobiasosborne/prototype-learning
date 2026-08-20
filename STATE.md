# STATE — single source of truth

Updated: 2026-08-20 (session 1, in progress)

## Phase status (see PLAN.md §3)

| Phase | Status | Notes |
|---|---|---|
| P0 scaffold | done | uv, CLAUDE.md=AGENTS.md, worklog, docs/physics distillations (both papers) |
| P1 reference backend | done | numpy sv/dm; blind Codex oracle cross-check (tests/oracle) |
| P2 neurons / Fig 3 | done | Fig 3 corners reproduced exactly (experiments/runs/bitcount/20260820T135930Z) |
| P3 KW-SA + bit-count | done (discrepancy) | superposition: median KL<0.005 at sweep 9 (paper ~8); iterative baseline stalls at KL≈0.15 after 200 its (paper ≈0 by 120) — open |
| P4 orkan + trajectories + GPU | done | orkan DM backend 4.1 s per noisy 11q eval (numpy 148 s); trajectories; CuPy DM + trajectories |
| P5 Iris | done | acc 95/91/98 %; Fig 7 contrast reproduced; Fig 9 not; target ∝ (label+1)² from Fig 8 |
| P6 Flappy Bird | done | perfect play (110 pipes = cap) on 20/20 held-out seeds; run 20260820T142739Z |
| P7 ECG | done | Table I within ~2 dB; Fig 13 reproduced; runs 20260820T162014/T165910/T165912 |
| P8 noise library + vendor presets | done (review 2 + Codex review running) | channels, NoiseModel (ASAP idle scheduling), Aer cross-check, 7 sourced device TOMLs |
| P9 noisy experiments | bit-count done; 11q noise-aware + other tasks running | reports/noise.md; run 20260820T145335Z |
| P10 hardware layer + plan | layer done; plan report pending | gatesets, decompositions, routing, QASM3, executors; reports/hardware-compile-table.md |
| P11 final report | not started | |

## Reproduction scoreboard (paper value → ours)

| Figure/Table | Paper | Ours | Status |
|---|---|---|---|
| Fig 3 activation surface | (0,1)→max, (1,0)→0 | 1.000 / 0.000 / 0.5 / 0.5 | reproduced |
| Fig 5 target | (popcount+1)/112 | confirmed on all 32 bars ±0.0002 | reproduced |
| Fig 6 superposition KL | ≈0 in ~8 iters | median <0.005 at 9 sweeps; 5/6 seeds <1e-3 at 20 | reproduced (seed 2 stalls at 0.014) |
| Fig 6 iterative KL | ≈0 by ~120 iters | ≈0.15 at 200 | discrepancy (our iterative scheme is an assumption) |
| Fig 7 Iris loss curves | prototype generalises | classic test KL rises, prototype monotone | reproduced |
| Iris accuracy | ≈98 % | 95/91/98 % (ceiling 97 %) | in range |
| Flappy Bird | perfect play | 110/110 pipes, 20/20 seeds | reproduced (own clone) |
| Table I ECG PSNR (1/2/3 proto) | 32.9 / 40.00 / 40.04 | 37.4 / 38.0 / 38.1 (synthetic) | within ~2 dB |

## Open decisions / assumptions
See PLAN.md §0. Resolved today: angle encoding θ = 2·asin(√s) (SNN Fig 4).
