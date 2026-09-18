# Codex review 004 — session-4 changes (adversarial, read-only)

You are reviewing a Python + LaTeX repository at /home/tobias/Projects/prototype-learning
(package `src/ptqnn`, tests in `tests/`, `uv run pytest -m "not slow"` runs the fast suite in ~2 min;
LaTeX report in `reports/tex/`). Review the diff `git diff afcfa48..HEAD` (session 4, today) as a
hostile referee. You have no conversation context; everything you need is below or in the repo.

## What the project is
A clean-room reproduction of "Prototype Learning for Quantum Neural Networks" (Rosenhahn & Hirche;
pinned distillation in docs/physics/rosenhahn_hirche_ptqnn.md) with a noise/hardware extension.
Conventions: qubit 0 = first input; prototype neuron = RX(φ_i) on each input, CRX(θ) from each
input to the hidden qubit, then RX(−φ_i) on each input (Fig 2); classic neuron = CRX(w_i) + bias.
Superposition training minimises KL(target ‖ model) with model(x) ∝ P(inputs = x, out = 1)
measured on the uniform superposition (Eq 4). Optimiser: Kiefer–Wolfowitz finite differences
inside simulated annealing (`src/ptqnn/train/kwsa.py`).

## Claims made this session that you must try to break
1. **Observable mismatch (src/ptqnn/tasks/bitcount.py, reports/tex/sections/04-reproduction.tex).**
   Claim: the prototype block does NOT restore the input register (RX(−φ) acts on a control that
   the CRX has entangled), hence P(inputs=x, out=1) ≠ 2^-n P(out=1|x); therefore the sample-wise
   ("iterative") scheme, which trains P(out=1|x) one basis input at a time, must be graded with the
   per-input distribution q(x) ∝ P(out=1|x) (`activation_distribution`, `perbasis_kl_loss`), and
   the old "stalls at KL≈0.15" was the superposition KL of a per-input-trained network. Verify the
   physics (write the 2-qubit case by hand: input |x⟩, RX(φ), CRX(θ) to hidden, RX(−φ); compute
   the reduced state of the input) and the code (`activation_distribution` really is P(out=1|x)
   renormalised; `basis_prep`; KL convention/epsilons consistent with `kl_loss`).
2. **New iterative baseline (`train_iterative`, `ITERATIVE_CONFIG`, `KWSAConfig.n_coords`,
   `probe_candidates`).** Check the schedule keying (`start_iteration = n // anneal_every`), that
   `probe_candidates=False` keeps the ±c probes for the derivative only, that with `n_coords=1` an
   iteration costs exactly 3 evaluations, that `monitor_every` indexing is documented and consistent
   with `experiments/bitcount_figs.py` (x-axis = index × monitor_every), and that the default
   `KWSAConfig` path is bit-for-bit unchanged. The run `experiments/runs/bitcount/20260918T111515Z`
   (histories.json, meta.json, results.md) backs the report numbers: seeds 1/2/3 reach per-input
   KL < 1e-3 at 73k/23k/12k iterations, seed 0 plateaus 0.02–0.03; superposition KL of the same
   networks 0.07–0.4. Re-derive these from histories.json and flag any number in 04-reproduction.tex,
   01-introduction.tex (scoreboard), A-assumptions.tex, reports/tex/overleaf/{sec-reproducibility,
   app-agentic}.tex that the data do not support.
3. **ECG mailed source (`src/ptqnn/tasks/ecg/data.py::load_mailed`, `train.prepare`,
   `experiments/ecg_table1.py`).** Files at data/raw/ecg-rosenhahn/*.mat (present). Check: no
   renormalisation, no added noise, sigma measured, `n_samples=None` handling, provenance digests,
   the run `experiments/runs/ecg/20260918T132924` (results.json, meta.json) vs the numbers in the
   new table `tab:ecg-mailed` (CQNN 33.97/36.33/38.02, PTQNN 36.80/37.48/36.86, input 27.0 dB) and
   the prose around it. Also: `meta.json` stores a NEGATIVE wall_seconds — find the bug in the
   script and state the one-line fix.
4. **Iris (`src/ptqnn/tasks/iris/{data,train}.py`, `experiments/iris_fig7_fig8_fig9.py`).**
   `drop_ambiguous_codes` (codes with ≥2 distinct labels removed from BOTH splits; is the test-set
   filtering defensible? does anything downstream assume the old counts?); `firing_matrix` claims an
   EXACT fast route via per-input-qubit Kraus factors of the hidden layer ("blocks on different input
   qubits commute") — verify the argument (the blocks RX(φ_i)·CRX_i→h·RX(−φ_i) for different i share
   the hidden target; controlled-X-rotations on one target with different controls commute — check
   carefully with the RX(−φ) un-computes present, and check the test that pins it to the reference
   backend exercises entangled cases); `dead_hidden_neurons` relative tolerance;
   `coverage_selectivity`; the `--rescore` mode.
5. **Report honesty.** In `reports/tex/sections/01-introduction.tex` ("Compute discipline") and the
   overleaf files: are the laptop/workstation provenance statements consistent with the run
   `meta.json` files present? Are any numbers stated without a run citation? Does
   `reports/tex/overleaf/sec-reproducibility.tex` claim anything the appendix does not support?

## Deliverable
Write NOTHING into the repo. Your final message is a Markdown review with sections:
BLOCKERS (wrong physics/maths/numbers; each with file:line, the failing argument or a snippet that
demonstrates it — you may run `uv run python -c ...` and `uv run pytest tests/<file> -q -m "not slow"`
to check, but keep any command under 3 minutes and do not run experiments/*.py),
MAJOR (correctness risks, missing tests, misleading prose), MINOR (style, naming, docs), and
VERDICT (one paragraph). Be specific and terse; no praise.
