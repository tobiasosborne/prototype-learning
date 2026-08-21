# Brief: adversarial review of session 2 (review round 3)

You are an independent reviewer.  You have no conversation context; everything you need is in
this brief and in the repository at the current directory.  Do not modify any file.  Report in
Markdown in your final message (it will be saved verbatim as `reviews/003-session2.md`).

## Project in one paragraph

`ptqnn` reproduces Rosenhahn & Hirche, *Prototype Learning for Quantum Neural Networks* (bit-count,
Iris, Flappy Bird, ECG) and hardens it with a CP-map noise layer built from vendor calibration
data, a hardware-compilation layer, and a "shot-noise programme".  Conventions: qubit 0 is the
most significant bit of a basis index; a 2-qubit gate matrix is indexed `2·b_q0 + b_q1` in the
order of `gate.qubits`; `CRX(θ) = |0⟩⟨0|⊗I + |1⟩⟨1|⊗RX(θ)`; every physics/ML fact in code is
pinned to a source (`[PTQNN Fig 2]`, `[SNN §III.A]`) or marked `[ASSUMPTION]`; tests must check
full distributions, not marginals; `uv run pytest -m "not slow"` is the fast suite.

## Scope — the session-2 diff

`git diff 7271620..HEAD --stat` is the scope (commits `ad720ab`, `c40c8e6`, `a44b3f6`, `6767bb6`
and any later ones).  The files that matter most:

1. `src/ptqnn/backends/batched.py` (+ `tests/test_batched_backend.py`): one batched statevector
   kernel that replaced three private `rx/crx` kernels in `tasks/ecg/model.py`,
   `tasks/flappy/agent.py`, `tasks/iris/model.py`.  Check the axis bookkeeping of
   `apply_gate_batched` for 2-qubit gates with `q0 > q1`, `basis_columns` bit order, and
   `product_input_states` placement of the input block (inputs are the *first* `n_inputs`
   qubits = high bits).  Try to construct a circuit on which it disagrees with
   `ptqnn.backends.reference.run_statevector`.
2. `src/ptqnn/backends/shots.py` (+ `tests/test_shots.py`): `ShotBackend` with common random
   numbers by *maximal coupling* inside `common_random_numbers()` blocks.  Verify that
   `coupled_counts` has marginal exactly `p_new` (it claims to) — derive it, and test it
   numerically with ≥ 10⁵ shots.  Check the claim that the expected number of changed shots is
   `N·TV`.  Check that nothing outside a block is affected and that `kwsa.minimise`'s
   `probe_context` wraps exactly the probes it should (`src/ptqnn/train/kwsa.py`).
3. `src/ptqnn/tasks/flappy/agent.py`: `SPRTDecision` (Wald's test on the sign of `p − ½`),
   `QuantumFlappyPolicy` with `shots`, `decision`, `frame_margins`, `activation_cache`.  Check
   the SPRT thresholds and the log-likelihood-ratio formula, the overshoot/cap behaviour, and
   whether `frame_margins`/`shots_used` are counted exactly once per frame.
   `src/ptqnn/tasks/flappy/train.py`: `margin_penalty` (per-frame hinge in units of the 0.01
   survival reward), `train(x0=…)`, and `train(batch > 0)` → `restate_incumbent=True` + final
   point returned.  Is the penalty scaling argument right (can a policy ever gain by dying)?
4. `experiments/hardware_plan.py` + `reports/hardware-plan.md` (+ run dir
   `experiments/runs/hardware-plan/20260821T073211Z/numbers.md`, gitignored but present on disk).
   Review 002 found the v1 report's numbers wrong; v2 claims every number comes from the script.
   Spot-check at least eight numbers in the report against `numbers.md` and against the
   device TOMLs in `src/ptqnn/noise/devices/`.  Check the fidelity-budget formulas
   (`idle_fidelity`, `fidelity_budget`), the SABRE critical-path walk
   (`_qiskit_critical_path_ns`), the state-prep CNOT counts, and the provenance section —
   does `section_provenance` really vouch for every number used, or can a field slip through
   (e.g. the `"snapshot epoch"` wildcard, or the readout duration)?
5. `experiments/bitcount_shots.py`, `experiments/flappy_shots.py` and their run dirs under
   `experiments/runs/{bitcount_shots,flappy_shots}/` (results.md): are the conclusions drawn in
   `reports/noise.md` ("shots" section) supported by the numbers?  Is anything over-claimed?
6. `reports/reproduction.md` (Flappy and Iris paragraphs changed), `reports/noise.md`.

## What to deliver

Findings ranked by severity (blocker / major / minor), each with file:line, what is wrong,
how you verified it (run code — `uv run python -c ...` and `uv run pytest <file>` are
available; the GPU needs `LD_LIBRARY_PATH=/usr/lib/wsl/lib` and may be skipped), and a concrete
fix.  Then a short "what checks out" list (things you tried hard to break and could not).  Be
specific and numeric; do not pad.  If a claim in a report or docstring is not supported by code
or data, say so.
