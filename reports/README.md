# reports/

Three Markdown reports and the LaTeX write-up. Every number cites a run directory under
`experiments/runs/` (see `experiments/README.md` for the pinned stamps) and is produced by a
script; prose is edited, numbers are regenerated.

| file | content | source of numbers |
|---|---|---|
| `reproduction.md` | every paper figure/table next to ours: Fig 3, 5, 6, 7, 8, 9, 11, 13, Table I; discrepancies explained, not tuned away | `bitcount_figs.py`, `iris_fig7_fig8_fig9.py`, `flappy_train.py`, `ecg_table1.py` |
| `noise.md` | the four tasks under seven vendor presets and swept strengths; noise-aware training (7 q); shot-noise bias; "Shots" section with the CRN / margin / SPRT results | `bitcount_noise.py`, `flappy_noise.py`, `ecg_iris_noise.py`, `bitcount_shots.py`, `flappy_shots.py` |
| `hardware-plan.md` (v2) | per-vendor path: logical vs compiled circuits, fidelity budgets, simulated TVD of compiled circuits, SABRE comparison, cost per shot, shot budgets, Iris state preparation, assumptions A1–A11 | `hardware_plan.py` → `runs/hardware-plan/20260821T084554Z/numbers.md` — the report quotes that file only |
| `hardware-compile-table.md` | gate set × topology cost matrix (depth, entanglers, SWAPs) for the Flappy and bit-count circuits | `hardware_compile_table.py`; do not edit |
| `tex/` | the final write-up, `main.pdf` (28 pp, 23 figures): intro, notation, models + optimiser, reproduction, noise, hardware, shot programme, learnings; appendices A (assumptions) and B (calibration provenance). Written at expert level, no jargon left undefined | figures from `report_figures.py` into `tex/figures/`; numbers from the run dirs above |

## Building the PDF

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib uv run python experiments/report_figures.py   # ≈ 1 min, 23 figures
cd reports/tex && latexmk -pdf main.tex
```

latexmk notes that bit us: `\ket` is predefined by quantikz (use `\renewcommand`); `\path`/`\url`
cannot sit inside captions; a global caption `width=` breaks side-by-side minipage figures.
Build artefacts (`.aux`, `.log`, …) are listed in `tex/.gitignore`; `main.pdf` is committed.

## `tex/overleaf/` — the drop-in for Bodo's Overleaf project

`tex/overleaf/` is a self-contained restructuring of this write-up for the PTQNN submission's own
Overleaf project (revtex4-2, `prx`, twocolumn), which had our sections 04/05/06/08 pasted in
near-verbatim and broken: our macros and four packages were missing, three labels dangled, and the
notation, shots, iterative-baseline, Iris-split and appendix material had been dropped.
It ships `ptqnn-agentic-preamble.tex` (booktabs/tabularx/siunitx, cleveref after hyperref, every
macro `\providecommand`-guarded, `\SI` handed back to his `ascii` package), `sec-reproducibility.tex`
(the ~0.6-column main-text subsection, with the scoreboard as a real LaTeX table instead of the
`PTAgentComp.png` screenshot, and the honest framing: simulators matched to vendor calibration data,
no hardware runs), `app-agentic.tex` (appendices A–I: notation, the four reproductions, calibrated
noise, hardware, shots, learnings, assumptions, provenance — every number and run-directory stamp
kept, 13 of our 23 figures), `figures-needed.txt`, `refs.bib`, `README-for-overleaf.md` and
`standalone.tex`, which rebuilds his preamble verbatim so the two inputs compile here
(`cd reports/tex/overleaf && latexmk -pdf standalone.tex`; 15 pp, no errors, no undefined
references). Nothing in it is used by `tex/main.tex`.

## Status and discrepancies

The scoreboard lives in `STATE.md`; the open discrepancies are the Fig 6 iterative baseline
(ours stalls at KL ≈ 0.15) and the Fig 9 TSNE picture. Grids not run for budget reasons (11-q
noise-aware bit-count, full ECG/Iris noise grids) are listed there with their estimated cost.
