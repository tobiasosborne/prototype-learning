# STATE — single source of truth

Updated: 2026-09-25 (end of session 5, workstation)

## Phase status (see PLAN.md §3)

| Phase | Status | Notes |
|---|---|---|
| P0 scaffold | done | uv, CLAUDE.md=AGENTS.md, worklog, docs/physics distillations (both papers) |
| P1 reference backend | done | numpy sv/dm; blind Codex oracle cross-check (tests/oracle); `backends/batched.py` shared kernel (session 2) |
| P2 neurons / Fig 3 | done | Fig 3 corners reproduced exactly (experiments/runs/bitcount/20260820T135930Z) |
| P3 KW-SA + bit-count | done (discrepancy) | superposition: median KL<0.005 at sweep 9 (paper ~8); iterative baseline stalls at KL≈0.15 after 200 its (paper ≈0 by 120) — open |
| P4 orkan + trajectories + GPU | done | GPU dispatch regression fixed (session 2) |
| P5 Iris | done | acc 95/91/98 %; Fig 7 contrast reproduced; Fig 9 not; target ∝ (label+1)² from Fig 8; split is an assumption (Fig 8 bin counts do not fix it) |
| P6 Flappy Bird | done | perfect play 110/110 on 20/20 held-out seeds; rotating seed batch = 25/25 once the incumbent is re-scored (session 2) |
| P7 ECG | done | Table I within ~2 dB; Fig 13 reproduced |
| P8 noise library + vendor presets | done; reviews 001/002/002b applied | channels, NoiseModel, Aer cross-check, 7 sourced device TOMLs |
| P9 noisy experiments | bit-count + Flappy done; ECG/Iris reduced run only; 11q noise-aware NOT run | reports/noise.md (incl. session-2 "Shots" section) |
| P10 hardware layer + plan | done; review-002 blocker 1 closed | `experiments/hardware_plan.py` → run dir + provenance; reports/hardware-plan.md v2 |
| P11 final report | **done** | `reports/tex/main.pdf` (28 pp, 23 figures from `experiments/report_figures.py`); reproduction.md, noise.md, hardware-plan.md |
| Shot-noise programme | items 1–3 done (1 negative, 2 partial, 3 positive); 4–6 not started | `backends/shots.py`, `kwsa.probe_context`, `flappy.agent.SPRTDecision`, margin objective |
| Review round 3 | closed: all 9 findings applied (reviews/003-session2.md resolution table) | |
| Inbox round (session 4) | **done (laptop)**; review 004 closed in session 5 | real ECG source + Table I run, Iris drop policy + placement metrics, iterative baseline recast (E2), `reports/tex/overleaf/` drop-in, reply draft |
| Workstation round (session 5) | **done** | review-004 code fixes; laptop runs regenerated from clean commits (bit-identical where they overlap); E2 six seeds; ECG full trace; Iris 20-seed × 2-policy sweep with pre-registered Fig 9 verdict; train-calibrated Iris read-out |
| Docs round (session 3) | **done** | `README.md`, `docs/{architecture,backends,noise,training,tasks,hardware,testing}.md`, READMEs in `experiments/`, `reports/`, `reviews/`, `notes/`, `tests/`, `data/`, `literature/`; no code changed |

## Reproduction scoreboard (paper value → ours)

| Figure/Table | Paper | Ours | Status |
|---|---|---|---|
| Fig 3 activation surface | (0,1)→max, (1,0)→0 | 1.000 / 0.000 / 0.5 / 0.5 | reproduced |
| Fig 5 target | (popcount+1)/112 | confirmed on all 32 bars ±0.0002 | reproduced |
| Fig 6 superposition KL | ≈0 in ~8 iters | median <0.005 at 9 sweeps; 5/6 seeds <1e-3 at 20 | reproduced (seed 2 stalls at 0.014) |
| Fig 6 iterative KL | ≈0 (≈1e-3) by ~120 iters | per-input KL < 1e-3 at 12k / 23k / 62k / 73k single-coordinate KW steps (seeds 3/2/4/1); seed 5 min 1.4e-3; seed 0 plateaus at 0.02–0.03; the same networks have superposition KL 0.035–0.58, endpoints 0.07–0.53 (run bitcount/20260925T153548Z, 6 seeds) | reproduced in the per-input observable; the iteration unit is an assumption (the old 0.15 was the superposition KL of a per-input-trained net) |
| Fig 7 Iris loss curves | prototype generalises | classic test KL rises, prototype keeps falling; 20 seeds, final > 1.05 × min: classic 8 (merge) / 7 (drop), prototype 2 / 0; prototype curves have small upticks on 9/20 | reproduced |
| Iris accuracy | ≈98 % | 95/91/98 % on the 3 August splits; 20 seeds (train-calibrated read-out): prototype 94.3 ± 2.8 %, classic 93.9 ± 2.7 % (merge); drop: 96.4 % filtered / 94.1 % all (run iris_sweep/20260925T175903Z). Set-refitted read-out is fragile (seed 18 drop: 39 % filtered vs 94 % all) | 2–4 points short; no architecture advantage |
| Fig 9 Iris TSNE | weights span data | t-SNE not reproduced; scored over 20 seeds × {merge, drop}, pre-registered criterion on z-scores vs each layer's own untrained null: fails (prototype selectivity z +17–18 vs classic +1.3–1.8, but coverage z +0.1–0.6 vs +1.9–2.2). Raw values: prototype covers more (0.80–0.89 vs 0.60–0.63; 19/20, 16/20 seeds) but untrained already 0.79 vs 0.39; classic more selective on all 40 runs (0.42–0.46 vs 0.14–0.15). Angle ρ: prototype CIs contain 0; classic direct/merge +0.14 [0.07, 0.22] (run iris_sweep/20260925T175903Z) | **discrepancy**: the spread is the architecture's, not training's |
| Flappy Bird | perfect play | 110/110 pipes, 20/20 seeds | reproduced (own clone) |
| Table I ECG PSNR (1/2/3 proto) | 32.9 / 40.00 / 40.04 | 37.4 / 38.0 / 38.1 (synthetic) | within ~2 dB |
| Table I ECG PSNR on the co-author's original recording (input PSNR 27.0 dB first 1800 / 27.5 dB full 7700; best of 8 restarts) | 32.9 / 40.00 / 40.04 (CQNN 32.7 / 38.2 / 39.65) | 1800: PTQNN 36.8 / 37.5 / 36.9, CQNN 34.0 / 36.3 / 38.0 (ecg/20260925T153739Z-mailed1800 = laptop run exactly); 7700: PTQNN 37.75 / 37.82 / 36.25 (36.2), CQNN 34.90 / 37.72 / 38.19 (ecg/20260925T165741Z-mailedfull) | 1-neuron advantage reproduced (larger than paper); 2/3-neuron cells 2–4 dB short, ordering at 3 flips — discrepancy; persists on the full recording under the same protocol |

## Shot-noise scoreboard (session 2)

| Remedy | Result | Run |
|---|---|---|
| Margin-aware training (per-frame hinge, m=0.1, w∈{0.25,0.5,1}) | margins unchanged (p50 0.019→0.021–0.024); play under shots worse at w≥0.5 — **negative** | flappy_shots/20260821T084601Z |
| Common random numbers (maximal coupling) for KW–SA | derivative variance ÷2.9 (c=0.2), ÷12–22 (c=0.05); sampled training 0.080→0.059 (N=1000), 0.050→0.014 (N=4000), still ≫ exact 0.0009 (plug-in bias) | bitcount_shots/20260821T084558Z |
| SPRT decisions (δ0=0.01, cap 10 000) | **100 % perfect at 2 191 shots/frame vs 10 000 fixed (÷4.6)** | flappy_shots/20260821T084601Z |

## Not run (compute ceiling 1–2 h per run)
* 11-q noise-aware bit-count grid (≈55 min/cell GPU, 14 cells); ECG/Iris full noise grids (4–8 h).

## Open decisions / assumptions
See PLAN.md §0 and reports/tex/sections/A-assumptions.tex.
