# Codex review 005 — session-5 changes (adversarial, read-only)

You are reviewing a Python + LaTeX repository at /home/tobiasosborne/Projects/prototype-learning
(package `src/ptqnn`, tests in `tests/`; `uv run pytest -m "not slow"` runs the fast suite in about
a minute; LaTeX report in `reports/tex/`, Overleaf drop-in in `reports/tex/overleaf/`). Review the
diff `git diff 5abedbe..HEAD` (session 5, today) as a hostile referee. You have no conversation
context; everything you need is below or in the repo. Read `reviews/004-inbox.md` first: this
session claims to have fixed every code finding of that review (resolution table at its end).

## What the project is
A clean-room reproduction of "Prototype Learning for Quantum Neural Networks" (Rosenhahn & Hirche;
distillation in docs/physics/) with a noise/hardware extension. Conventions: qubit 0 = first input
= most significant bit; prototype neuron = RX(φ_i) on each input, CRX(θ) from each input to the
hidden qubit, then RX(−φ_i) on each input; classic neuron = CRX(w_i) from each input + RX(bias).
Iris: 12-bit one-hot k-means code per flower, 12 inputs → 4 hidden → 1 output (17 qubits); the
class label ℓ enters as target weight ∝ (ℓ+1)²; labels are read out by rounding each code's
read-out value to the nearest of the levels 1:4:9 after fitting one overall scale.

## Claims made this session that you must try to break
1. **Review-004 fixes are real** (commits a2dd8da, 6c8857f, 4915fa2). For each item in the
   resolution table of `reviews/004-inbox.md`, check the code and that a test pins it and would
   have failed before (read the test against the parent commit's code where cheap).
   In particular: `iris.train.hidden_register_state` vs the reference partial trace (is the test
   non-vacuous: entangled/non-product cases, both kinds?); `IrisSplit` index invariant;
   `--rescore` refusal logic in `experiments/iris_fig7_fig8_fig9.py`; `bitcount.monitor_x`;
   `bitcount_figs.py` final vs best bookkeeping.
2. **Regenerated runs back the report.** Runs (all present under experiments/runs/):
   `bitcount/20260925T153548Z` (6 seeds; claims: seeds 3/2/4/1 cross per-input KL 1e-3 at
   12k/23k/62k/73k steps, seed 5 min 1.4e-3, seed 0 plateau 0.02–0.03, superposition KL of the same
   nets 0.035–0.58 with endpoints 0.07–0.53; superposition series identical to
   `bitcount/20260820T135930Z`), `ecg/20260925T153739Z-mailed1800` and
   `ecg/20260925T165741Z-mailedfull` (tables in 04-reproduction.tex `tab:ecg-mailed`), the Iris
   sweep `iris/20260925T*-seed*-{merge,drop}` (40 dirs) summarised by
   `experiments/iris_sweep_summary.py` into `iris_sweep/20260925T173530Z`. Re-derive every number
   quoted in reports/tex/sections/{01-introduction,04-reproduction,08-learnings,A-assumptions}.tex,
   reports/tex/overleaf/{app-agentic,sec-reproducibility}.tex, STATE.md and
   notes/reply-to-bodo-draft.md from these files; flag any that the data do not support.
3. **Fig 9 verdict (`experiments/iris_sweep_summary.py`).** The criterion is stated in its
   docstring and in `iris.train.firing_placement` / `prototype_placement`. Check the statistics:
   one-sided paired Wilcoxon direction, bootstrap CI of the median, the "rises" rule for the Fig 7
   contrast, that nothing silently skips a run, and that the report's wording ("prototype layer has
   the selectivity and not the coverage", "angle score zero within error") follows from
   summary.json.
4. **The read-out finding (`iris.model.calibrated_labels`, `calibrated_accuracy`).** Claim: the
   original read-out (`labels_from_values` on the evaluated set) makes a code's label depend on the
   other evaluated codes; on sweep seed 18 (drop) the prototype model scores 39 % on the 87
   filtered test samples but 94 % on all 100, and with the scale fitted once on the training codes
   96.6 %. Verify by computation from `experiments/runs/iris/*seed18-drop/params-prototype.npy`.
   Is calibrating on the training set sound (does it leak test information? is the normaliser
   Z = Σ_train P(out=1|x) the right one given how `fit_level_scale` normalises)? Is the claim that
   with `codes = calib_codes` it equals `labels_from_values` of the per-sample read-out true?
   Note the two read-outs differ in kind: `accuracy` uses the superposition over the evaluated
   set's codes (`regressed_values`), `calibrated_accuracy` the per-sample basis activations — is
   the report's comparison of them fair, and does it say which is which?
5. **Report honesty.** Compute-provenance paragraph (01-introduction.tex), the ECG paragraph's
   "rules out the window", and the new learnings item.

## Deliverable
Write NOTHING into the repo. Your final message is a Markdown review with sections: BLOCKERS
(wrong physics/maths/numbers; each with file:line and a demonstrating snippet — you may run
`uv run python -c ...` and `uv run pytest tests/<file> -q -m "not slow"`, each under 3 minutes;
do not run experiments/*.py except `iris_sweep_summary.py` with `--out` pointing into /tmp),
MAJOR (correctness risks, missing tests, misleading prose), MINOR, and VERDICT (one paragraph).
Be specific and terse; no praise.
