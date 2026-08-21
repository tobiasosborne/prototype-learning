# ptqnn — prototype quantum neural networks under realistic noise

A reproduction of Rosenhahn & Hirche, *Prototype Learning for Quantum Neural Networks*
(PTQNN; bit-count, Iris, Flappy Bird, ECG denoising) and of the optimiser from its prerequisite,
Rosenhahn, Osborne & Hirche, *Stochastic Neural Networks for Quantum Devices*
(SNN, [arXiv:2602.22241](https://arxiv.org/abs/2602.22241)) — then hardened: a CP-map noise layer
fed by vendor calibration data, noisy and finite-shot training, transpilation to native gate sets,
and a costed plan for running on real hardware.

The written deliverable is `reports/tex/main.pdf`. Where things stand: `STATE.md`
(scoreboard), then `HANDOFF.md` (work queue). Design and conventions for agents: `CLAUDE.md`
(mirrored as `AGENTS.md`).

## Install and test

```bash
uv sync --group dev                 # Python 3.12 venv
uv run pytest                       # fast suite (~30 s, ≈440 tests); add -m slow for long runs
sh scripts/check-all.sh             # ruff + ruff format + mypy --strict + fast pytest
```

Optional extras: `uv sync --extra qiskit` (Aer cross-check, IBM ISA dry run),
`--extra gpu` (CuPy; on WSL2 export `LD_LIBRARY_PATH=/usr/lib/wsl/lib`), `--extra ecg` (wfdb for
MIT-BIH). The orkan C backend needs `../orkan/cmake-build-release/src/liborkan.so`
(`uv run pytest -m orkan`).

## Thirty-second tour

```python
import numpy as np
from ptqnn.circuits.neurons import Network
from ptqnn.tasks import bitcount
from ptqnn.backends.reference import StatevectorBackend
from ptqnn.noise.presets import device
from ptqnn.noise.model import NoisyBackend
from ptqnn.backends.reference import DensityMatrixBackend

net = bitcount.network(n_bits=5, n_hidden=5, hidden="prototype")      # 11 qubits [PTQNN Fig 5]
res = bitcount.train_superposition(net, seed=1, iterations=20, backend=StatevectorBackend())
print(res.loss)                                                        # KL(target ‖ model) ≈ 5e-4

noisy = NoisyBackend(DensityMatrixBackend(), device("ibm_fez").noise_model())
print(bitcount.kl_loss(net, noisy)(res.params))                        # same loss under vendor noise
```

Every training loop depends only on the `Backend` protocol (`probabilities(circuit)`), so the
same code runs on the exact statevector, a density matrix with CP-map noise, Monte-Carlo
trajectories, the GPU, a finite-shot sampler, or (dry-run) a hardware executor.

## Layout

| Path | What |
|---|---|
| `src/ptqnn/` | the package — `circuits`, `backends`, `noise`, `train`, `tasks`, `hardware` (`docs/architecture.md`) |
| `tests/` | pytest suite; `tests/oracle/` is a blind second simulator used as a cross-check (`docs/testing.md`) |
| `experiments/` | one script per experiment, each writing `experiments/runs/<name>/<stamp>/` with `meta.json` (`experiments/README.md`) |
| `reports/` | `reproduction.md`, `noise.md`, `hardware-plan.md`, and the LaTeX report in `tex/` (`reports/README.md`) |
| `docs/` | architecture, backends, noise, training, tasks, hardware, testing; `docs/physics/` holds the pinned distillations of both papers |
| `reviews/` | independent review rounds and their resolutions (`reviews/README.md`) |
| `notes/briefs/` | self-contained briefs handed to Codex |
| `worklog/`, `WORKLOG.md` | session log shards and index; `PLAN.md` is the original plan with its assumption register |
| `literature/`, `data/` | local-only PDFs/figure crops and the MIT-BIH record (gitignored; see their READMEs) |

## Reproduce the paper's figures

```bash
uv run python experiments/bitcount_figs.py                 # Fig 3, 5, 6         (~2 min)
uv run python experiments/iris_fig7_fig8_fig9.py           # Fig 7, 8, 9         (~10 min)
uv run python experiments/flappy_train.py --seed 0         # Fig 10/11 agent     (~10 min)
uv run python experiments/ecg_table1.py --source synthetic # Table I, Fig 13     (~10 min)
LD_LIBRARY_PATH=/usr/lib/wsl/lib uv run python experiments/report_figures.py   # all report figures
cd reports/tex && latexmk -pdf main.tex
```

Headline status (details in `STATE.md`): Fig 3, 5, 6 (superposition), 7, Flappy perfect play and
Table I (±2 dB) reproduced; the Fig 6 iterative baseline and the Fig 9 TSNE picture are recorded
discrepancies, not tuned away.

## Conventions that matter

* Qubit 0 is the most significant bit of every basis index and the first input; the paper's
  1-based top-to-bottom numbering is ours +1.
* Every circuit op is a CP map (`Gate` with a unitarity witness, `Channel` in Kraus form).
* Every physics/ML fact in code is pinned to a source: `[PTQNN Fig 2]`, `[SNN §III.A]`; our own
  choices are marked `[ASSUMPTION]` and listed in `PLAN.md §0` / `reports/tex/sections/A-assumptions.tex`.
* Backends are checked against the numpy reference on full distributions, never on marginals.

License: MIT. Principal: Tobias J. Osborne.
