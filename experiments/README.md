# experiments/ — scripts and run directories

One script per experiment. Each writes `experiments/runs/<name>/<UTC stamp>/` containing at
least `meta.json` (seed, git hash — with a `-dirty` suffix if the tree was not clean — backend
name, every CLI option, wall time) plus machine-readable results (`results.json`, `.npz`/`.npy`)
and its figures. Reports cite run dirs by stamp; `report_figures.py` reads only the
machine-readable files (never a PNG) from the stamps pinned in its `RUNS` dict. `runs/` is
gitignored (≈ 11 MB today); the stamps cited in `STATE.md` and the reports are the ones to
regenerate if they are lost.

## Scripts

| script | produces | paper pin | cost (this box) |
|---|---|---|---|
| `bitcount_figs.py` | Fig 3 surface, Fig 5 histograms, Fig 6 convergence (6 seeds, superposition + iterative) | Fig 3, 5, 6 | ~2 min |
| `iris_fig7_fig8_fig9.py` | Fig 7 loss curves, Fig 8 histograms, Fig 9 TSNE, accuracies; classic + prototype in two processes | Fig 7–9 | ~10 min (17-q statevector) |
| `flappy_train.py` | trained agent `params.npy`, `log.csv`, `best.gif`, held-out scores | Fig 10, 11 | ~3–10 min |
| `ecg_table1.py` | Table I grid (2 kinds × 1/2/3 hidden), Fig 13; `--source synthetic|physionet` | Table I, Fig 13 | ~10 min, one process per cell |
| `bitcount_noise.py` | studies A (deploy noisy per preset), B (noise-aware training), C (strength sweep), D (shots) on the Fig-5 net; `--studies`, `--devices`, `--backend` | — | A/C/D ~15 min on GPU; B at 11 q **not run** (≈ 55 min/cell) |
| `flappy_noise.py` | trained agent played under each preset, scaled noise, finite shots; optional noisy fine-tune | — | ~5 min |
| `ecg_iris_noise.py` | ECG cells and Iris under presets/scales/shots; Iris via trajectories sized to `--iris-budget-s`; optional ECG fine-tune | — | reduced config ~10 min; full grids 4–8 h, not run |
| `bitcount_shots.py` | KW–SA on `ShotBackend` with/without common random numbers, per N | shot programme item 2 | ~2 min |
| `flappy_shots.py` | margin-aware fine-tune (item 1) and SPRT decisions (item 3), measured by playing | items 1, 3 | ~11 min |
| `hardware_compile_table.py` | gate set × topology cost table → `reports/hardware-compile-table.md` | — | seconds |
| `hardware_plan.py` | every number of `reports/hardware-plan.md` (`numbers.md`/`.json`), provenance join | — | ~2 min on GPU (`--no-sim` for seconds) |
| `report_figures.py` | all 23 figures of `reports/tex/` as PDF + PNG from the pinned `RUNS` | — | ~1 min |
| `iris_sweep_summary.py` | aggregates Iris seed-sweep run dirs into the pre-registered Fig 9 verdict, both accuracies, both read-outs (`--calibrated`), Fig 7 contrast | HANDOFF item 2 | ~25 s with `--calibrated` on 32 procs |

All accept `--out` and a seed; `--help` lists the rest. GPU-capable scripts need
`LD_LIBRARY_PATH=/usr/lib/wsl/lib` on WSL2 to see the RTX 3090.

## Pinned run directories (what the reports quote)

| run | stamp | commit | used by |
|---|---|---|---|
| bitcount | `20260820T135930Z` | 04727b9 | reproduction.md Fig 3/5/6, tex §4 |
| iris | `20260820T144849Z`, `20260820T145454Z`, `20260820T145455Z` (seeds 0–2) | 453d9ae / 54c79eb | reproduction.md, tex §4, hardware_plan state-prep |
| flappy | `20260820T143753Z` (also `batch0`, `batch5` for the rotating-batch check) | e6ebd59 | reproduction.md, flappy_noise, hardware_plan |
| ecg | `20260820T162014` (synthetic), `20260820T165912` (PhysioNet 100) | 54c79eb / e25e31a | reproduction.md Table I, tex §4 |
| bitcount_noise | `20260820T145335Z` (+ `…-B11/results.md`: the 11-q cell estimate) | 871ee9e | noise.md, tex §5 |
| flappy_noise | `20260820T154249Z` | 44c1e0b | noise.md |
| ecg_iris_noise | `20260820T155618Z` | a330088 | noise.md (reduced run) |
| bitcount_shots | `20260821T084558Z` | 2d88102 | noise.md "Shots", tex §7 |
| flappy_shots | `20260821T084601Z` (`20260821T075048Z` is the failed mean-hinge attempt, kept as evidence) | 2d88102 | noise.md "Shots", tex §7 |
| hardware-compile-table | `20260820T155816Z` | 44c1e0b | hardware-compile-table.md |
| hardware-plan | `20260821T084554Z` | 2d88102 | hardware-plan.md, tex §6 |
| bitcount (E2, iterative baseline) | `20260925T153548Z` (6 seeds × 1e5 elementary steps; seeds 0–3 = laptop run `20260918T111515Z` bit for bit) | a2dd8da | tex §4 Fig 6 (both panels) |
| ecg (mailed recording) | `20260925T153739Z-mailed1800` (= laptop `20260918T132924` in every cell), `20260925T165741Z-mailedfull` (7700 samples) | 6c8857f | tex §4 `tab:ecg-mailed` |
| iris (sweep) | `20260925T*-seed{0..19}-{merge,drop}` (40 dirs; launcher/logs in `iris/sweep-4915fa2/`) | 4915fa2 | iris_sweep |
| iris_sweep | `20260925T173530Z` (`--calibrated`) | a80ad50 | tex §4 Iris, STATE scoreboard |
| codex | `NN-events.jsonl`, `NN-err.log` | — | raw Codex transcripts behind `reviews/` and `tests/oracle/` |

## Rules

* **Budget.** Nothing longer than 1–2 h is launched; estimate from a measured per-evaluation
  cost first and print it (`flush=True`). Grids that would exceed it are recorded as "not run"
  with the estimate (`STATE.md` "Not run"). Log background runs to a file, not through `tail`.
* **Provenance in the file, not the shell history.** Seed, git hash, backend and every option go
  in `meta.json`; a run whose tree was dirty says so.
* **Reports quote, scripts compute.** A number in `reports/*.md` or `reports/tex` that changes
  is changed by re-running the script and regenerating — never by editing the prose
  (`hardware_plan.py` docstring; review 002 blocker 1 is why).
* **Scripts are thin.** Physics and ML live in `src/ptqnn`; a script builds configs, loops,
  records and plots. Their pure helpers are unit-tested in `tests/test_*_script.py`.
* **Figures** follow one palette (`report_figures.py`: prototype blue, classic orange, paper
  grey, one hue per device) and are regenerated, not edited.
