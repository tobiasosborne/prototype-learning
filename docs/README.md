# docs/ — how the code is put together

Read in this order if you are new: `architecture.md` → whichever layer you touch → `testing.md`.
The paper itself is distilled, with equation/figure pins, in `physics/`.

| File | Covers |
|---|---|
| `architecture.md` | package map, the CP-map circuit IR, qubit/bit-order conventions, data flow from paper figure to run directory |
| `backends.md` | the six `Backend` implementations, their limits and agreement tests; GPU/orkan environment |
| `noise.md` | Kraus channel library, `NoiseModel` scheduling semantics, vendor presets and provenance, readout, scaling |
| `training.md` | the KW–SA optimiser and its `[ASSUMPTION]` hyper-parameters, losses, the superposition-training observable, finite-shot training (CRN, SPRT) |
| `tasks.md` | the four tasks (bit-count, Iris, Flappy Bird, ECG): what the paper fixes, what we chose, module map, run dirs |
| `hardware.md` | transpilation passes, native gate sets, OpenQASM bit order, executors, cost model, `hardware_plan.py` |
| `testing.md` | TDD rules, markers, gates, the blind oracle, statistical tolerances, how to add a test |
| `codex.md` | how Codex (gpt-5.6-sol) is invoked for blind implementations and adversarial review |
| `physics/rosenhahn_hirche_ptqnn.md` | PTQNN distillation: every figure, equation and quoted sentence the code pins to |
| `physics/rosenhahn_osborne_hirche_snn.md` | SNN (arXiv:2602.22241) distillation: optimiser, angle convention, Iris binarisation |

Pins used throughout code and docs: `[PTQNN Fig 2]`, `[PTQNN Eq 4]`, `[SNN §III.A]` refer to the
section numbers of the two distillations above. `[ASSUMPTION]` marks a choice the papers leave
open; the register is `PLAN.md §0` and `reports/tex/sections/A-assumptions.tex`.

Project-state documents live at the top level, not here: `STATE.md` (scoreboard), `HANDOFF.md`
(queue), `PLAN.md`, `WORKLOG.md` + `worklog/`, `CLAUDE.md` (= `AGENTS.md`).
