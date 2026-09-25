# reviews/ — independent review rounds

Review is part of every phase, not an afterthought. After a phase goes green an independent
reader — an Opus subagent or Codex (`gpt-5.6-sol`, xhigh, read-only sandbox; `docs/codex.md`) —
reads the diff against `PLAN.md` and the paper pins for correctness, API design, test adequacy
and readability. Findings are filed here; each is either fixed (with the test that pins the
fix) or explicitly declined with the reason, in a **Resolution** table at the bottom of the
file. The main session integrates only after re-running the tests itself.

| round | scope | reviewer | headline findings | status |
|---|---|---|---|---|
| `001-core-sim-and-training.md` | P1–P3: IR, reference backend, orkan FFI, KW–SA, losses, bit-count; commit `a7b58ca` | Opus subagent | iteration-unit ambiguity (Fig 6 reported in sweeps *and* evaluations), aliasing of caller arrays, test gaps | closed |
| `002-tasks-backends-hardware.md` | P4–P10: noise layer, fast backends, tasks, hardware; commit `871ee9e` (moving target, HEAD advanced during review) | Opus subagent | **blocker 1**: hardware plan cited no run dir and a dozen numbers were wrong → `hardware_plan.py`; #6 test side effect → `hardware_compile_table.py`; #7 three private statevector kernels → `backends/batched.py`; #8 stale incumbent under rotating seeds → `restate_incumbent`; #9 Iris split wording | closed (session 2) |
| `002b-codex-noise-adversarial.md` | noise library + trajectories; brief `notes/briefs/codex-02-noise-review.md` | Codex | 3 defects: `ReadoutError` validation, `TrajectoryBackend(batch_size=0)` infinite loop, `scaled(1)` altering generic channels; Choi/Kraus, T2>T1 vs Aer, depolarising/fidelity, ASAP scheduling confirmed | closed |
| `003-session2.md` | session-2 diff `7271620..HEAD`; brief `notes/briefs/codex-03-session2-review.md` | Codex | **blocker**: Iris state-prep budgeted for 120 samples instead of 12–14 distinct codes (9× over-count; "Iris out of reach" withdrawn); provenance validator matched substrings not values; `-dirty` hashes; CRN scaling stated as 1/c² not 1/c; idle relaxation double-counted; wording "paper schedule"; margin scaling; SNR arithmetic; lint | closed — all 9 applied, re-runs at a clean commit |
| `004-inbox.md` | session-4 diff `afcfa48..a21b734`; brief `codex-04-session4-review.md` | Codex | iterative cost 4 not 3 evaluations; `final_kl` stored the minimum; negative ECG wall time; Iris drop filter conditions the test set; firing factorisation tested on marginals only; 200-draw null | closed (session 5; code fixes `a2dd8da`, `4915fa2`) |
| `005-session5.md` | session-5 diff `5abedbe..afc13fe`; brief `codex-05-session5-review.md` | Codex | **blocker**: Fig 9 z-scores read as a between-architecture comparison (raw values: prototype covers more but already untrained; classic more selective); classic angle CI excludes 0; reply-draft iteration arithmetic; provenance over-claim; Fig 7 monotonicity wording | closed (session 5) |

Raw transcripts of the Codex rounds: `experiments/runs/codex/NN-events.jsonl` (gitignored; the
final `agent_message` is what was saved here verbatim).

## Writing a brief

Briefs live in `notes/briefs/`. Codex has no conversation context, so a brief inlines every
convention it needs (qubit 0 = MSB, `RX = exp(−iθX/2)`, Choi convention, file paths), the exact
deliverable, and — for reviews — the commit range and a request to *try to break* specific claims
with numerical checks. The "What checks out" section of a review is as valuable as the findings:
it records which claims were independently verified and how.

## Next round

Per `CLAUDE.md`, the next review follows the next green phase (shot-noise programme items 4–6,
`HANDOFF.md`). Number it `004-<phase>.md`; never renumber.
