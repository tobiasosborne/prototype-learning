# Review 003 — session 2 diff (Codex gpt-5.6-sol, xhigh, read-only), 2026-08-21

Brief: notes/briefs/codex-03-session2-review.md. Events: experiments/runs/codex/03-events.jsonl.

# Review 003 — session 2, round 3

Scope reviewed through `50426be`, including the uncommitted Flappy training changes explicitly named in the brief. No files were modified.

## Blocker

### 1. The hardware report budgets the wrong Iris state and cannot support “Iris is out of reach”

Locations: `experiments/hardware_plan.py:456-473`, `reports/hardware-plan.md:247-258`, `reports/tex/sections/06-hardware.tex:119-121`, `reports/tex/sections/A-assumptions.tex:42-44`.

The sparse estimate hard-codes `k_train=120`, but this repository’s Iris runs train on 50 samples and prepare a uniform superposition over distinct codes, not samples. The cited runs have only 14, 12, and 14 nonzero training amplitudes:

- `experiments/runs/iris/20260820T144849Z/meta.json:21-27`
- corresponding seed-1/2 runs: 12 and 14 distinct codes
- `src/ptqnn/tasks/iris/model.py:78-93` explicitly deduplicates with `np.unique`

Using the report’s own assumed leading term `k·n`, the preparation count is therefore 144–168, not 1,440. At H2-1’s error, I recomputed:

- preparation two-qubit factor: 0.831–0.853, not 0.205
- including the network’s 104 entanglers: 0.741–0.761

The unknown constant in `O(k·n)` still prevents a positive feasibility claim, but the reported “best case ≈0.2” and resulting “Iris is out of reach” conclusion do not follow.

There is also a citation error: Shende–Bullock–Markov gives at most `2^(n+1)−2n = 8168` CNOTs for `n=12`, not `2^(n+1)−2n−2 = 8166`; the latter is associated with another construction. See the [NIST-hosted Shende–Bullock–Markov paper](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=150894). The Plesch–Brukner `23/24·2^n` figure is correctly described as a leading-order upper bound ([primary paper](https://harvest.aps.org/v2/journals/articles/10.1103/PhysRevA.83.032302/fulltext)).

Fix: derive the support from the exact Iris run/split used by the report, distinguish samples from distinct basis amplitudes, correct the dense attribution, and withhold a hardware-fidelity conclusion until an actual sparse preparation circuit supplies an exact gate count.

## Major

### 2. `section_provenance` does not validate provenance despite the report’s fail-closed claim

Locations: `experiments/hardware_plan.py:479-496`, `experiments/hardware_plan.py:564-620`, `reports/hardware-plan.md:359-365`, `reports/tex/sections/B-provenance.tex:4-10`.

The check is presence-based substring matching; it never compares the used value with the provenance entry’s value. Additional holes:

- `"snapshot epoch"` is a wildcard for every nonzero IBM field.
- `"gate_1q"` lets a duration entry vouch for an error, and similarly for two-qubit/readout fields.
- `native_1q`, which changes gate-set selection, is not checked.
- The passed `devices` values are ignored: line 568 reloads each raw preset by name.
- Synthetic heavy-hex/square patches and non-native Google/Rigetti rebasing are recorded in `CompiledCircuit.notes`, but `numbers.md` drops those notes and A1–A9 omit these assumptions.

I injected arbitrary IBM calibration values with only the `snapshot epoch` provenance entry. `section_provenance` returned `unsourced=[]` and treated that one entry as the source for every field. A second injection showed an IonQ `gate_1q.duration_ns=130000` entry vouching for an arbitrary `gate_1q_error=0.987654`.

The shipped IBM readout duration does have a real explicit source; the defect is that the validator cannot establish that fact.

Fix: use exact canonical field identifiers, compare scalar/list values, validate the actual passed `DeviceSpec`, remove the epoch fallback, include `native_1q`, and render every compilation assumption. Add mutation tests proving that a changed value or removed exact entry fails.

### 3. The shot runs’ recorded git hashes do not identify their source, and the Flappy training summary compares different objectives

Locations: `experiments/bitcount_shots.py:54-60`, `experiments/flappy_shots.py:66-72`, `src/ptqnn/tasks/flappy/train.py:304-332`, `experiments/runs/flappy_shots/20260821T075321Z/results.md:3`.

Both shot runs record hash `6767bb6`, but neither experiment script exists at that commit (`git cat-file -e 6767bb6:<script>` exits 128). More seriously, the recorded commit’s Flappy trainer has neither `x0` nor the per-frame penalty used to produce the run. Unlike `hardware_plan.py`, these scripts do not record a dirty suffix.

The Flappy summary’s loss arrows also compare different episode horizons:

- `init_loss` is evaluated with `restart_steps=250`
- `final_loss` uses the actual 1,500-step training objective

Recomputing the same 1,500-step objective gives:

| weight | reported arrow | comparable 1,500-step arrow |
|---:|---:|---:|
| 0.25 | −5.97 → −44.20 | −43.97 → −44.20 |
| 0.5 | −5.48 → −41.43 | −40.96 → −41.43 |
| 1.0 | −4.49 → −35.63 | −34.96 → −35.63 |

Finally, `train(x0=...)` skips 300 restarts but still adds `n_restarts` to `evaluations`. Each run reports 1,576 evaluations; the actual training-objective count is `1 + (1 + 25·3·17) = 1,277`.

Fix: commit the exact source before rerunning or refuse dirty runs; record a dirty/diff hash; evaluate `init_loss` on the main objective when `x0` is supplied; and count one cheap `x0` evaluation rather than skipped restarts.

### 4. The CRN theory and the claim that estimator bias—not optimisation—sets the result are not supported

Locations: `src/ptqnn/backends/shots.py:7-28`, `src/ptqnn/train/kwsa.py:121-125`, `tests/test_shots.py:71-101`, `reports/noise.md:82-100`, `reports/tex/sections/07-shots.tex:75-105`.

For nearby categorical distributions, maximal coupling changes a shot with probability `TV=O(c)`. Therefore:

- variance of the unscaled sample-mean difference is `O(c/N)`
- after division by `2c`, gradient variance is `O(1/(cN))`

It is not `O(c²/N)` as the Python docstrings claim, nor `O(c/N)` for the slope as the TeX report claims. The empirical `1/c` improvement over independent `O(1/(c²N))` gradients is consistent with the correct result.

The samples have correct marginals, but the plug-in KL is nonlinear and biased, so its finite difference is not unbiased for the exact-KL slope. At the report’s `c=0.2`, 400 shots, and 5,000 repetitions, I measured:

- exact slope: 0.32231
- maximally coupled mean: 0.34557
- standard error: 0.00271
- bias: 8.6 standard errors

Thus `tests/test_shots.py:99-101` is underpowered and `reports/tex/sections/07-shots.tex:95` is false.

The run data also contradict “the sampled-training floor sits 1–3× above [the plug-in bias].” Exact-final-KL/bias ratios for off/on CRN are:

- N=1,000: 2.5 / 1.9
- N=4,000: 6.3 / 1.8
- N=16,000: 14.4 / 16.6

With only three seeds—and an exact-probability seed already stalled at KL 0.014—the experiment cannot separate optimiser failure from estimator bias.

Fix: correct the scaling derivation; describe CRN as variance reduction, not debiasing; test against the derivative of the expected sampled objective; and compare biased versus corrected/joint losses under paired initialisations and substantially more seeds before assigning causality.

### 5. The hardware fidelity budget charges full-path relaxation as “idle” on top of vendor gate errors

Locations: `experiments/hardware_plan.py:165-203`, `reports/hardware-plan.md:127-165`, `reports/tex/sections/06-hardware.tex:32-47`.

`idle_fidelity` itself is the correct average fidelity for a thermal-relaxation channel. The problem is its use:

```text
F_idle = F_avg(total critical-path time)^n_qubits
```

This charges every qubit for the entire path, including periods when it is executing gates, after already multiplying by vendor RB gate survival factors that include relaxation. A circuit with every qubit continuously active still receives an “idle” penalty.

Using the compiled schedules and charging only rough per-wire gaps changed:

- Fez Flappy idle factor: 0.684 → 0.782
- Fez bit-count: 0.265 → 0.419
- IQM Flappy: 0.382 → 0.511

The CP-map simulator acknowledges related double counting in `src/ptqnn/noise/model.py:15-20`, but the report presents these products as expected fidelities rather than pessimistic bounds.

Fix: either apply relaxation only to scheduled idle gaps while treating vendor gate errors as total gate errors, or subtract the coherence-limited contribution before composing gate relaxation. Regenerate the fidelity rankings and label remaining approximations explicitly.

## Minor

### 6. Project-chosen optimiser hyperparameters are repeatedly called the paper’s schedule

Locations: `experiments/bitcount_shots.py:7-8`, `reports/noise.md:82-83`, `reports/tex/sections/07-shots.tex:76-78,96-97`, `reports/tex/sections/06-hardware.tex:76`.

`src/ptqnn/tasks/bitcount.py:90-96` correctly marks `a0=20`, `c0=0.2`, exponents, cooling, and 20 sweeps as `[ASSUMPTION]`; the paper publishes none of them. Replace “paper schedule,” “paper’s c,” and “paper’s 2,160 evaluations” with “our assumed reproduction schedule/budget.”

### 7. The margin scaling is safe only for the default return component, not the full configurable objective

Locations: `src/ptqnn/tasks/flappy/train.py:95-124`, `src/ptqnn/tasks/flappy/train.py:168-172`.

For default `step_reward=0.01` and `weight<1`, the return-plus-hinge contribution of an additional frame is indeed favorable by at least `0.01·(1−weight)`. However:

- `0.01` is hard-coded despite a configurable `env_config.step_reward`.
- The full objective includes a non-additive mean tracking-error term that can outweigh 0.01.
- The experiment includes `weight=1`, for which a zero-margin extra frame merely ties.

Use `env_config.step_reward`, constrain/document the allowed weight, and narrow “always worth more” to the return-plus-margin component.

### 8. A hand-calculated hardware SNR is wrong, and static device noise does not generally cancel

Location: `reports/hardware-plan.md:288-291`.

The report’s own expression gives:

```text
c·|∂p/∂θ|·2√N = 0.2·0.5·2√4000 = 12.65
```

not 25. For two independent binomial probes the corresponding difference SNR is about 17.9.

Only a parameter-independent additive loss offset cancels in a central difference. A fixed bit-flip channel, for example, maps `p → e+(1−2e)p` and rescales the derivative by `1−2e`; static device noise can therefore bias the gradient.

Fix the arithmetic and qualify cancellation to common additive drift/offsets.

### 9. The session’s required lint gate is red

Location: `experiments/report_figures.py:208,455,790,832,1062`.

`ruff check src tests experiments` reports five errors: two unused loop variables, two overlong lines, and one unused `noqa`. `ruff format --check` and `mypy src` pass.

## What checks out

- Batched evolution agreed with the reference in 770 adversarial state/circuit cases over every ordered qubit pair, including generic asymmetric two-qubit unitaries with `q0>q1`; worst amplitude error was `1.67e-16`. Reversed `basis_columns` order `(3,0,2)` and high-bit `product_input_states` placement were exact.
- The maximal coupling is correct. Algebraically, the retained mass is `min(p_ref[i],p_new[i])` and the residual adds `(p_new[i]−p_ref[i])₊`, giving marginal `p_new`; rejected outcomes have total probability `TV`. At one million shots, empirical moved fraction was `0.280168` for `TV=0.28`, and the largest marginal deviation was 1.12 standard deviations.
- CRN state is cleared in `finally`; calls outside a block are ordinary independent samples. In coordinate mode, `probe_context` wraps exactly the `+c`, `−c`, and proposed-step evaluations, not the initial or incumbent restatement.
- SPRT likelihood ratio and thresholds are correct for valid `alpha,beta`. Boundary simulations at `p=0.48/0.52` gave approximately 3.4–5.3% error depending on block size; awkward `max_shots=37, block=20` stopped at exactly 37. `frame_margins` and `shots_used` increment once per `act`, hence once per played frame.
- `train(batch>0)` correctly restates the incumbent and returns the final accepted point when the batch genuinely rotates.
- `_qiskit_critical_path_ns` reproduced the recorded O1–O3 paths; the actual ISA circuits contain only `sx/x/rz/cz`, so its treatment of measurement/readout is correct for these runs.
- More than eight hardware figures were independently reconciled against `numbers.md` and the TOMLs, including Flappy logical `7q/d14/12 CRX/19 1q`; H2 Flappy `24 entanglers/d36`; Fez Flappy `90/22 swaps/d297/12.888 µs`; IQM bit-count `207/49/d485/18.130 µs`; all four H2 Flappy fidelity factors; Fez SABRE O2 `37 CZ/100 physical 1q/d98/F=0.718745`; `P(out=1)=0.487091`; KL coefficient `31.8216`; and the IBM/IonQ/IQM cost rows.
- Full non-slow suite: `435 passed, 3 skipped, 6 deselected` in 28.43 s. Mypy passed for all 40 source files.
