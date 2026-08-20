# ptqnn — prototype quantum neural networks with realistic noise

Reproduce Rosenhahn & Hirche, *Prototype Learning for Quantum Neural Networks* (bit-count,
Iris, Flappy Bird, ECG), then harden it: a CP-map noise layer fed by vendor calibration data,
noisy training runs, and a path to real hardware. Principal: Tobias Osborne (expert; write at
expert level, no fluff). Plan: `PLAN.md`. Where things stand: `STATE.md`, then `HANDOFF.md`.
This file is mirrored as `AGENTS.md`.

## Build & test

```bash
uv sync --group dev                 # Python 3.12 venv, ~10 s
uv run pytest                       # fast suite; add -m slow for the long runs
uv run pytest -m orkan              # needs ../orkan/cmake-build-release/src/liborkan.so
uv run ruff check . && uv run mypy src
sh scripts/check-all.sh             # all of the above; run before every worklog entry
```

## Ground truth discipline

- Every physics/ML fact that enters code is quoted *locally* in the docstring or a comment,
  pinned to a source: `[PTQNN Fig 2]`, `[PTQNN Eq 4]`, `[SNN §III.A]` (SNN = arXiv:2602.22241).
  Distillations with the pinned quotes live in `docs/physics/*.md`; the PDFs in
  `literature/pdf/` are local only (gitignored, copyright). If a fact is our choice because the
  paper is silent, the comment says `[ASSUMPTION]` and `PLAN.md §0` lists it.
- Red → green TDD. The failing test is written first and committed with the fix; a test that
  never failed proves nothing. Statistical tests use ≥1000 samples and a stated tolerance.
- Test channels, not marginals. A teleportation that moved only the diagonal once passed a
  marginal test in a sister project; here, backend agreement is checked on full density
  matrices / Choi matrices, never only on P(out = 1).
- Figures read from the paper carry the reading error (e.g. Fig 5 bar heights ±0.001). Paper
  numbers we cannot match are reported as discrepancies, never tuned away.

## Gotchas

- Paper qubit numbering is 1-based top-to-bottom in the circuit figures; code is 0-based with
  qubit 0 = first input. Bit strings in the Fig 5 histogram are MSB = qubit 1 (first input).
- The prototype neuron un-computes its input rotations (`RX(-φ)` after the CRX, Fig 2); the
  classic neuron does not touch inputs. Mixing them up silently still "trains".
- Model distribution in superposition training is P(inputs = x, out = 1) renormalised over x,
  not P(out = 1 | x). KL(target ‖ model), target first (Eq 4).
- The optimiser is Kiefer–Wolfowitz finite differences inside simulated annealing
  (SNN §III.A). No hyper-parameters are published; ours are in `ptqnn/train/kwsa.py` and
  marked `[ASSUMPTION]`.
- orkan (`../orkan`) has no CRX, no sampling, no RNG, 1-qubit Kraus only, density matrices
  cap at ~14 qubits (2 GB). The ctypes binding mirrors `state_t` byte-exactly; see
  `../Sturm.jl/src/orkan/ffi.jl` for the working Julia precedent.
- `../QuantumHardware.jl/devices/*.toml` is thin: no T1 data anywhere, Quantinuum H1/H2 are the
  only devices with gate fidelities + readout. Vendor presets here record their own provenance.

## Conventions

- Package `src/ptqnn/`: `circuits` (IR, neurons), `backends` (numpy reference is the oracle;
  `orkan`, `trajectories`, `cupy` must agree with it), `noise` (Kraus/CP maps, `NoiseModel`,
  vendor presets), `train`, `tasks`, `hardware`.
- Every experiment is a script in `experiments/` that writes to `experiments/runs/<name>/`
  with the seed, git hash, and backend in a `meta.json`. Reports in `reports/` cite run dirs.
- Review rounds are part of every phase, not an afterthought: after a phase goes green, an
  independent reviewer (Opus subagent and/or Codex) reads the diff against `PLAN.md` and the
  paper pins for correctness, API design, test adequacy and readability; findings are filed in
  `reviews/NNN-<phase>.md`, fixed or explicitly declined there, then the phase is closed.
- Subagents: Opus/Sonnet only (no Fable). Codex (`gpt-5.6-sol`, xhigh) for blind second
  implementations and adversarial review — see `docs/codex.md`. Subagents report; the main
  session integrates after re-running the tests itself.

## Session close

1. `sh scripts/check-all.sh` green (or the failure recorded verbatim).
2. Update `STATE.md`; rewrite `HANDOFF.md` if priorities moved.
3. Add the session entry at the top of the newest `worklog/` shard (200–500 lines per
   shard, newest first; format in `worklog/README.md`); update the `WORKLOG.md` index.
4. Commit (conventional commits). Never renumber sessions.
