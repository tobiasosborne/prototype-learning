# Review 001 — core simulator and training (P1–P3)

Reviewer: independent Opus subagent. Date: 2026-08-20. Commit under review: `a7b58ca`
(scope files verified unchanged at `HEAD` = 47955f4 via `git diff --stat a7b58ca HEAD -- <scope>`).

Scope: `src/ptqnn/circuits/{ir,neurons}.py`, `src/ptqnn/backends/{base,reference,_orkan_ffi}.py`,
`src/ptqnn/train/{kwsa,losses,distribution}.py`, `src/ptqnn/tasks/bitcount.py`, `tests/*.py`,
`tests/oracle/indep_sim.py`.

Method: everything below was **executed**, not read off. Throwaway scripts under `/tmp`
(`chk1..chk11.py`, `prof*.py`); no repository file was modified except this one.

## Summary of what checks out

Before the findings, the things I tried hard to break and could not:

* **Gate sequences match the paper.** Fig 1 in `literature/QPrototypeLearning-5.txt` prints
  `|q3⟩ Rx(θ) Rx(ϕ1) Rx(ϕ2)` with `q1`, `q2` as controls — exactly `classic_neuron`
  (`rx(out, bias)` then `crx(in_i, out, w_i)`). Fig 2 prints
  `|q1⟩ Rx(ϕ1) … Rx(−ϕ1)`, `|q2⟩ Rx(ϕ2) … Rx(−ϕ2)`, `|q3⟩ Rx(θ) … Rx(θ)` — exactly
  `prototype_neuron`, one shared θ, no separate bias. Fig 5 (image) shows 11 wires, 5 H on the
  inputs, five prototype blocks each reading all 5 inputs, and the output bias `RX` on wire 11
  *before* the five CRX; the code emits the bias after the hidden layer, which commutes (the
  hidden layer never touches the output qubit) — equivalent, worth a one-line comment.
* **CRX convention**: `|0⟩⟨0|⊗I + |1⟩⟨1|⊗RX(θ)`, control = first listed qubit, verified against
  explicit `kron` embeddings for `(c,t)` in both orders and against the blind oracle.
* **Qubit ordering is consistent across all four implementations.** `basis_prep(0b10000)` puts
  the weight at flat index 1024 of the 11-qubit register (qubit 0 = MSB). Reference vs blind
  oracle on the full Fig-5 circuit: `max|Δ| = 3.1e-17`. Reference vs **orkan** (11 q,
  `MIXED_TILED`, gates applied at the same integer indices, read back with
  `to_numpy(order="msb0")`): `max|Δ| = 1.5e-17`; the induced model distributions agree to
  4.2e-17. A hand-written marginal that decodes bit strings by hand agrees with
  `ref.marginal` to 0.0.
* **Model distribution and KL direction.** `conditioned_input_distribution` = `P(inputs=x,
  out=1)` renormalised over `x` = `P(x | out=1)` (same object), and `kl_loss` calls
  `kl_divergence(target, q)` — target first, `[PTQNN Eq 4]`. Confirmed numerically that this is
  *not* the same as renormalising the per-basis-state `P(out=1|x)` (`max|Δ| = 1.7e-2` for a
  random 5-hidden network): the prototype block does not restore the input register (the CRX
  entangles the control), so the superposition histogram genuinely differs from the sample-wise
  one. The code implements the Fig-5 quantity. Good.
* **`marginal` axis handling is right**, including unsorted `keep`: `marginal(p, [2,0])` returns
  `[0, 0, 0.5, 0.5]` where `[0,2]` returns `[0, 0.5, 0, 0.5]` for `|+⟩|0⟩|1⟩`. `tensordot` +
  `moveaxis` in `_apply_matrix_to_axes` was checked against `kron` embeddings for random
  unsorted qubit pairs: error 0.0.
* **The `[ASSUMPTION]` hyper-parameter provenance reproduces.** The comment at
  `tasks/bitcount.py:88-91` claims "median KL after 12 sweeps 1.1e-3; worst seed 1.5e-2" over 6
  seeds. I ran it: median `1.15e-3`, worst `1.51e-2`. Determinism for a fixed seed confirmed.
* **Paper quotes are real.** Every quoted passage I checked exists in
  `literature/QPrototypeLearning-5.txt`, modulo the two-column extraction interleaving text
  mid-sentence (so a literal substring search fails but the wording is verbatim). Notably the
  `[PTQNN Fig 2]` quote in `neurons.py:10-11` ("increasing/decreasing") matches the §III.B body
  text, not the Fig 2 caption (which says only "increasing") — the body is the better pin.
* Full suite green including `-m slow` (123 tests); `ruff` and `mypy --strict` clean on the
  reviewed files.

No blockers. Eight majors, mostly about what will hurt in P4/P5/P9/P10 rather than about the
physics.

---

## Findings

### 1. `train_superposition` silently ignores `seed` unless `iterations` is passed — major

`src/ptqnn/tasks/bitcount.py:104-108`

```python
cfg = config or SUPERPOSITION_CONFIG
if iterations is not None:
    cfg = kwsa.KWSAConfig(**{**cfg.__dict__, "iterations": iterations, "seed": seed})
x0 = net.random_params(np.random.default_rng(seed))
```

`seed` is threaded into `cfg` **only** on the `iterations is not None` branch. Called as
`train_superposition(net, seed=s)` the initial point varies with `s` but the optimiser RNG is
always `SUPERPOSITION_CONFIG.seed == 0`; passing an explicit `config=` has the same hole.
Evidence (`/tmp/chk5.py`, 3-bit net):

```
train_superposition(net, seed=1)                 -> loss 1.2e-05
train_superposition(net, seed=1, iterations=20)  -> loss 7.12e-04     identical: False
```

Two calls that a reader would expect to be the same run are not. For a repo whose convention is
"every experiment records its seed", a seed sweep produced by the short call form shares one
Metropolis/permutation stream across all seeds.

**Fix**: build the config once, unconditionally —
`cfg = dataclasses.replace(config or SUPERPOSITION_CONFIG, seed=seed, **({"iterations": iterations} if iterations else {}))`
— and add a test asserting `train_superposition(net, seed=1) != train_superposition(net, seed=2)`
and `== train_superposition(net, seed=1)`. (`dataclasses.replace` also removes the `__dict__`
splat, which breaks the day `KWSAConfig` gets `slots=True`.)

### 2. "Frozen" `Gate`/`Channel` alias shared, writable matrices; the unitarity witness is defeatable — major

`src/ptqnn/circuits/ir.py:35-36, 48-58, 132-135, 177`

`_as_complex` calls `np.ascontiguousarray`, which is a **no-op returning the same object** for an
already-contiguous `complex128` array. So `ir.h(0).matrix is ir._H` — every `h` gate in the
process shares one module-level array, as do `x`/`y`/`z`, and `unitary(qs, m)` aliases the
caller's array. Evidence (`/tmp/chk2.py`):

```
shared module array: True  (g2.matrix is g1.matrix: True)
g1.matrix[0,0] = 99.0
ir.h(5)  ->  ValueError: matrix for gate 'h' is not unitary
```

One stray in-place write (a noise model scaling a matrix, a transpiler pass in P10, a user
poking at `.matrix`) permanently poisons the gate factory *and* invalidates the unitarity
witness that `__post_init__` advertises. `frozen=True` protects the field binding, nothing else.

**Fix**: in both `__post_init__`s, copy and freeze — `m = np.array(m, dtype=np.complex128);
m.setflags(write=False)` — and store `kraus` as a `tuple` of read-only arrays. Add a test:
`with pytest.raises(ValueError): ir.h(0).matrix[0, 0] = 1`.

### 3. `Gate`/`Channel` equality and hashing raise — major

`src/ptqnn/circuits/ir.py:39, 73` (dataclass-generated `__eq__`/`__hash__` over ndarray fields)

```
ir.rx(0,0.3) == ir.rx(0,0.3)  -> ValueError: truth value of an array ... is ambiguous
hash(ir.rx(0,0.3))            -> TypeError: unhashable type: 'numpy.ndarray'
g in [other_gate]             -> ValueError
```

So gates cannot go in a `set`, be dict keys (the obvious cache for finding 5 below), or be
compared in a test. It already weakens one: `tests/test_neurons.py:103`

```python
assert circ2.ops[:2] == prep or [o.name for o in circ2.ops[:2]] == ["h", "h"]
```

passes only because list `==` short-circuits on *identity* of the same `prep` objects; with
distinct-but-equal gates the first clause would raise rather than fall through to the `or`.

**Fix**: `@dataclass(frozen=True, eq=False)` plus an explicit `__eq__`/`__hash__` (name, qubits,
params, and `matrix.tobytes()` or `np.array_equal`), or drop `eq` and compare structurally in
tests. Then rewrite `test_network_circuit_structure` to assert on `(name, qubits, params)`
tuples and delete the `or`.

### 4. `KWSAResult` mixes best-ever and last-accepted semantics; `train_iterative` picks the other one — major

`src/ptqnn/train/kwsa.py:97, 117, 130, 132, 135`; `src/ptqnn/tasks/bitcount.py:133-136`

`minimise` returns `params`/`loss` = **best point ever seen**, while `history` records the
**current** point after each iteration (which under `t0 > 0` may be uphill). They differ:

```
res.loss (best) = 0.22382   res.history[-1] (last accepted) = 1.19903
f(res.params) == res.loss   -> True
```

`train_iterative` then constructs `KWSAResult(params=x, loss=history[-1], …)` — i.e. *last*, the
opposite convention, and its `history` is a KL trajectory while `loss` in `minimise` is whatever
`f` was. Fig 6 will be plotted from `history` while the reported model is `params`; the two need
not correspond, and the slow test (`res.history[-1] < 3e-3`) is asserting on the trajectory, not
on the returned model. Related: `train_iterative` sets `x = res.params` (best of the inner
sweep), which discards the SA state and quietly turns the inner anneal into greedy descent.

**Fix**: make the contract explicit — `KWSAResult(params, loss)` = the point you should use,
plus `best_params/best_loss` *and* `history` documented as the accepted-state trajectory
(and `best_history` if Fig 6 should show the monotone curve). Make `train_iterative` obey the
same convention. One test asserting `f(res.params) == res.loss` and
`res.history[-1] == f(<accepted state>)` would pin it.

### 5. `Backend.probabilities` returning a dense `2**n` vector will not survive P5/P9/P10 — major

`src/ptqnn/backends/base.py:17-23`

The protocol has exactly one method returning the full basis distribution. Measured cost of the
dense object alone (`/tmp` script, background run):

```
n=19: p = 4 MiB   (statevector 8 MiB)    marginal 5 ms
n=22: p = 32 MiB  (statevector 64 MiB)   marginal 35 ms
n=25: p = 256 MiB (statevector 512 MiB)  marginal 270 ms
```

Iris is 19–25 qubits (PLAN §0) and every loss evaluation would allocate and immediately discard
that, to keep 13 qubits' worth of marginal. Worse, the protocol has no notion of **shots**
(P9 needs shot noise), no **rng**, no **noise model**, and no way for a hardware `Executor`
(P10) — which returns counts over a *measured subset* — to implement it honestly. `distribution.
output_activation` (`train/distribution.py:47-52`) already builds `2**n` floats to read one
number.

**Fix now, while there is one implementation pair**: widen the protocol to
`probabilities(circuit, qubits: Sequence[int] | None = None)` (backends compute the marginal
themselves; the reference can keep using `ref.marginal`) and add
`sample(circuit, shots, rng, qubits=None) -> counts`. Backends that can only sample implement
`probabilities` as `None`-returning/`NotImplementedError`, and training code asks for what it
needs. Also move `marginal` out of `backends.reference` — `train.distribution` importing the
reference *backend* for a pure array utility is a layering inversion that will force every
backend to depend on the reference module.

### 6. The hot loop rebuilds and re-validates 81 gates per loss evaluation — major (performance)

`src/ptqnn/circuits/neurons.py:126-143`, `src/ptqnn/circuits/ir.py:57`

Profile of one 11-qubit bit-count KL evaluation (`/tmp/prof.py`, `/tmp/prof2.py`; the box was
under load ≈40, so read the *proportions*, the absolutes were 3.7× larger at load 100):

```
one KL evaluation (11 qubits): 5.14 ms
  build circuit  3.27 ms  (64 %)   <- Network.circuit -> 81 Gate constructions
  simulate       1.82 ms  (35 %)   <- 86 ops, 21 us/op of tensordot/moveaxis/reshape
  marginal + KL  0.02 ms  ( <1 %)

cProfile, sorted by tottime (200 evaluations):
  0.287 s  numpy.isclose        <- Gate.__post_init__ unitarity witness
  0.199 s  numpy.tensordot
  0.129 s  ir.py:48 __post_init__
build with the unitarity witness bypassed: 3.27 ms -> 0.42 ms
```

So **~55 % of every loss evaluation is `np.allclose(U†U, I)` on 81 gates whose unitarity is a
theorem** (`_rot` and `controlled` cannot produce a non-unitary), and a 20-sweep run is
20 × 36 × 3 = 2160 of those.

**What a 10× would need** (measured, not guessed):
1. Stop re-validating factory-built gates — validate in `unitary()`/`Gate` only when the matrix
   comes from outside, or cache by `(name, params)`. Free 2.2× overall.
2. Stop rebuilding the circuit: `Network.circuit` should produce a parametrised circuit once and
   the loss should update only the 81 `theta`s. Together with (1) this leaves the 1.82 ms sim.
3. Batching the finite-difference probes over a leading axis helps but saturates on cache:
   `B=1: 1.72 ms/circuit, B=3: 0.93, B=6: 0.74, B=12: 1.65, B=72: 1.53` — best ≈2.3×.
   (1)+(2)+(3) ≈ 6×. The last factor has to come from taking the 86-op loop out of Python
   (numba, or an orkan pure-state path — orkan already does a 12-qubit *density-matrix* gate in
   8.6 ms where numpy needs ~90 ms at 9 qubits). I tried a hand-rolled einsum/slice kernel
   (`/tmp/prof3.py`): 5.12 ms, i.e. **slower** — don't bother micro-optimising numpy here.

### 7. The reference density-matrix backend cannot carry the noisy bit-count runs — major

`src/ptqnn/backends/reference.py:57-62, 138-144` (docstring claims "≤ 12 qubits in practice")

```
n= 6 ops= 24: dm      6.7 ms | sv  0.90 ms | ratio    7
n= 8 ops= 44: dm   1575.2 ms | sv  4.41 ms | ratio  358
n= 9 ops= 57: dm  10334.8 ms | sv 44.04 ms | ratio  235
n=11 (Fig 5 circuit, 86 ops): did not finish in 300 s
```

(loaded box; the scaling ≈8× per qubit is the robust part). One noisy KL evaluation at the
project's headline size is therefore minutes, and P9 wants thousands of them. I also tried the
obvious rewrite — treat ρ as 2ⁿ stacked statevectors and contract the ket and bra sides with two
batched `tensordot`s (`/tmp/chk11.py`): identical results, **1.2–1.3×**, not worth it.

**Fix**: keep the numpy DM path as the *oracle* (it is exactly that, and the blind cross-check
is excellent), cap it explicitly at ~9 qubits in the docstring, and route P9's 11–12-qubit noisy
runs to orkan (`MIXED_TILED`, already bound and verified above) or trajectories. Add a
`pytest.mark.slow` benchmark that records the per-gate cost so the regression is visible.

### 8. The Fig-6 reproduction claim rests on a redefinition of "iteration" — major (paper fidelity)

`src/ptqnn/train/kwsa.py:16-20`

> "Empirically this reproduces the '≈0 in under 10 iterations' of `[PTQNN Fig 6]`; whole-vector
> KW steps (`mode="full"`) need O(100) iterations on the same loss."

One `mode="coordinate"` iteration is a full sweep: `3 × n_params = 108` loss evaluations for the
Fig-5 network. So "9 iterations" is ~1000 circuit evaluations, whereas the paper's iteration is
one KW–SA update. Both halves of Fig 6 use the same unit here (`train_iterative` also runs one
sweep per iteration), so the *comparison* is fair — but the sentence as written claims a match
to the paper's iteration count that is unit-dependent, and the second clause is not reproducible:

```
coordinate, 12 iters, 1297 evals -> best 9.67e-04
full,      108 iters, 7885 evals -> best 7.79e-02      (same a0=20, alpha=0.3, c0=0.2)
full,      100 iters, 7301 evals -> best 5.12e-02  (a0=1.0, alpha=0.602, c0=0.2)
                                 -> best 1.14e-02  (a0=5.0)
                                 -> best 7.64e-02  (a0=0.3, c0=0.1)
```

`mode="full"` does not reach ≈0 in O(100) iterations at any setting I tried, at 5.6× the
coordinate run's evaluation budget.

**Fix**: state the unit ("one iteration = one coordinate sweep = 3·n_params evaluations") and
report Fig 6 against `KWSAResult.evaluations` as well as iterations — the field already exists.
Correct the `mode="full"` sentence to what was measured (it stalls near 1e-2), and record this
in the reproduction report as a deviation, not a match. STATE.md's honesty about the iterative
baseline (0.15 vs ≈0) is the right template.

### 9. `conditioned_input_distribution` returns a *plausible* distribution for a dead network — minor

`src/ptqnn/train/distribution.py:41-44`

When `P(out=1) == 0` the function returns the uniform distribution, so the loss reports
`KL(target ‖ uniform) = 0.065` — a value indistinguishable from a mediocre-but-alive network,
and a discontinuity in the landscape the SA has to cross. Verified with an all-zero-parameter
2-input network.

**Fix**: return `+inf`/a documented large penalty, or better `np.full(..., nan)` and let the
caller decide; at minimum raise the `[ASSUMPTION]` to the docstring of `kl_loss` too.

### 10. `kl_divergence` accepts unnormalised, negative and NaN inputs silently — minor

`src/ptqnn/train/losses.py:17-24`

```
kl(p=[.5,.5], q=[5.,5.])   -> -2.303      (negative "divergence")
kl(p=[.5,.5], q=[-1., 2.]) ->  12.78      (negative mass clipped to eps)
kl(p=[.5,.5], q=[nan,.5])  ->  nan        (propagates into the optimiser)
kl(p=[1.,0.], q=[0.,1.])   ->  27.63      (the eps floor caps KL, flattening the landscape)
```

A NaN reaching `accept()` makes `delta <= 0` false and `rng.random() < exp(-nan)` false, so the
optimiser silently stops accepting instead of failing. `cosine_dissimilarity` returns `nan` on a
zero vector (`RuntimeWarning`, `losses.py:31`).

**Fix**: validate in the loss (`sum ≈ 1`, `q >= 0`, `np.isfinite`) behind a cheap assert, or at
least document that inputs must be normalised; make `kwsa.accept` raise on a non-finite delta.

### 11. `Literal` types are not enforced at runtime — minor

`src/ptqnn/circuits/neurons.py:137-141`, `src/ptqnn/train/kwsa.py:54`

```python
Network(2, 1, hidden='protoype').circuit(...)   # typo
-> ['rx', 'crx', 'crx', 'rx', 'crx']            # silently a CLASSIC network
KWSAConfig(mode='coordinat').mode -> 'coordinat'  # falls through to the "full" branch
```

This is precisely the CLAUDE.md gotcha ("mixing them up silently still trains") made reachable
by a typo, and mypy will not catch a string that arrives from a config file or CLI in P9.

**Fix**: validate in `__post_init__` (`if self.hidden not in get_args(HiddenKind): raise`),
and dispatch on an explicit mapping rather than `if/else`.

### 12. Untested surfaces — minor

* `train_iterative` has **no test at all**. Smoke-tested here (3 inputs, 2 hidden, 25 iterations):
  KL `3.01e-01 → 2.52e-01` (851 evaluations) versus `3.01e-01 → 8.46e-04` for superposition at
  826 evaluations — qualitatively Fig 6, and consistent with the STATE.md discrepancy note. It
  should have at least this smoke test so the Fig-6 red curve cannot silently break.
* No test drives an `ir.Circuit` through **orkan** and compares to `backends.reference`. The
  orkan tests are excellent but check orkan against a *locally written* kron reference, so the
  two conventions could drift apart without any test noticing. I did the check (11 q, agreement
  1.5e-17); it belongs in `tests/` behind `-m orkan`.
* No test for `marginal` with unsorted `keep` (it is correct — pin it), for duplicate `keep`
  (raises an opaque "axes don't match array"), or for `keep` given as a generator (raises).
* No determinism/seed test for `train_superposition` (would have caught finding 1).
* No test for the 6-hidden (Fig 4, 12-qubit) variant.
* `test_random_network_kl_is_order_of_fig6_start` accepts `0.005 < median < 0.5`, a two-decade
  band — it would pass under several wrong normalisations. Consider asserting the median against
  a stored value with a tolerance instead.
* `Circuit(0)` and `Circuit(-2)` are accepted without complaint; `sample_counts` on an all-zero
  `p` raises numpy's `pvals contains NaNs` rather than a domain error.

### 13. `_orkan_ffi` documented "system loader" fallback is dead code — minor

`src/ptqnn/backends/_orkan_ffi.py:193, 299`

`load_orkan`'s docstring promises "then the system loader", but the candidate returned by
`ctypes.util.find_library("orkan")` on Linux is a bare soname (`liborkan.so`), and the loop
guards every candidate with `os.path.exists(path)` — a soname on the loader path fails that test
and is skipped. Either resolve it (`ctypes.CDLL(found)` in a `try`) or drop the claim.

Everything else in that file is unusually careful: the ABI assertion, the `TILE_DIM` probe, the
column-major/row-major notes, the `two_from_mat` argument-order trap and its `SWAP4` fix, and
the "orkan `exit()`s instead of returning an error" guards are all correct as far as I could
exercise them, and `to_numpy(order="msb0")` composes with native qubit indices to give exactly
the project convention (verified in finding 0/summary above — no index remapping needed, which
deserves an explicit sentence in the module docstring since it is surprising).

### 14. Smaller API / hygiene points — nit

* `Network.unpack` (`neurons.py:112-124`) returns **views** into the caller's array; mutating a
  returned angle array mutates the parameter vector.
* `ir.Circuit.__post_init__` mutates `initial_state` in place and `append` re-validates nothing
  about `initial_state`; `Circuit` is a mutable dataclass whose `__eq__` will raise like
  finding 3.
* `Gate.dagger()` (`ir.py:67-70`) names the inverse of a non-parametrised gate `"h†"`, so
  `Circuit.count("h")` misses it and `dagger().dagger()` gives `"h††"`.
* `neurons.py:3-4` quotes Fig 1 as "…(CRX) gates. A plain rotation realizes the bias." The paper
  reads "…(CRX) gates and a plain rotation realizes the bias" (§II.B); the caption sentence is
  different again. Stitching two sources into one quotation is exactly what the ground-truth
  discipline is meant to prevent — split it into two pins.
* `marginal`'s `q not in keep` membership test is O(n·|keep|) and silently mis-handles
  duplicates; take a `frozenset` for the drop computation and validate `len(set(keep)) ==
  len(keep)`.
* `SUPERPOSITION_CONFIG` is a module-level mutable-by-convention default shared by
  `train_superposition` *and* `train_iterative` (`bitcount.py:121`), where the iterative scheme
  has no evidence that these hyper-parameters are appropriate — the grid search cited in the
  comment was run on the superposition loss only. Give the iterative baseline its own config
  (even if identical) so the provenance comment stays true.

---

## Verdict

**Accept with required fixes.** The physics is right: gate sequences, CRX convention, the
`P(x, out=1)`-renormalised model distribution, KL direction, and MSB-first qubit ordering all
check out against the paper text, the figures, a blind oracle and the orkan C library, at
1e-17 agreement on full density matrices. The ground-truth discipline is real — I could not find
an invented quote, and the one documented grid-search claim reproduced to two digits. The blind
Codex oracle is the best thing in the tree.

The weaknesses are all forward-looking: the `Backend` protocol (5), the per-call circuit rebuild
(6) and the density-matrix kernel (7) are load-bearing decisions that get expensive to change
after P4/P5/P9 land on top of them, and they should be fixed now rather than in P11. Findings 1
(seed) and 4 (best-vs-last) are reproducibility bugs that will silently corrupt the figures
before anyone notices, and 8 needs a sentence changed in the reproduction report rather than any
code. Findings 2 and 3 are latent but cheap to fix while `ir.py` has exactly two consumers.

Suggested order: 1, 4, 8 (correctness of the record) → 5, 6 (before more backends exist) → 2, 3,
11 → 12 (tests) → the rest.

---

## Resolution (main session, 2026-08-20)

| # | Decision | Where |
|---|---|---|
| 1 | Fixed: `train_superposition` always threads `seed` into the optimiser config. | `bitcount.py`, `test_seed_controls_the_whole_run` |
| 2 | Fixed: `_as_complex` copies and sets `writeable=False`; factories no longer alias constants. | `ir.py`, `test_gate_matrices_are_private_and_read_only` |
| 3 | Fixed: `Gate`/`Channel` define value `__eq__`/`__hash__` (`eq=False` dataclasses). | `ir.py`, `test_gate_equality_and_hash` |
| 4 | Fixed: `KWSAResult` contract documented — `params/loss` = best, `final_*` = accepted state, `history` = accepted trajectory, `best_history` = running min; `train_iterative` resumes from the accepted state and returns the same shape. | `kwsa.py`, `bitcount.py`, `test_kwsa_result_contract_best_vs_final` |
| 5 | Declined for now: every planned experiment is ≤ 17 qubits (Iris: 12 in + 4 hidden + 1), i.e. ≤ 1 MiB per call; executors returning counts are a separate `Executor` protocol (P10). Revisit if Iris grows beyond 20 qubits. | — |
| 6 | Fixed the dominant term: factory gates skip the unitarity witness (`_trusted`), `unitary()`/direct `Gate(...)` still verify. KL eval 5.1 ms → 3.7 ms under the same load; circuit rebuild per call kept (simplicity) — parametrised circuits are a follow-up if training budgets demand it. | `ir.py` |
| 7 | Accepted: noisy runs use the orkan (C, OpenMP) and CuPy (RTX 3090) density-matrix backends and the trajectory backend; the numpy DM backend stays the ≤ 9-qubit oracle. | `backends/orkan.py` (in progress), `backends/gpu.py` |
| 8 | Fixed wording: one iteration = one coordinate sweep = `3·n_params` evaluations; Fig 6 is reported in sweeps *and* evaluations; the `mode="full"` claim replaced by what was measured. | `kwsa.py` docstring, reports |
| 9 | Fixed: dead network → zero vector → KL penalty ≈ 27.6 instead of KL(target‖uniform). | `distribution.py`, `test_dead_network_is_penalised` |
| 10 | Fixed: `kl_divergence` validates shape, finiteness, normalisation. | `losses.py`, `test_kl_rejects_garbage` |
| 11–14 | Minor/nit: deferred to the next round (tracked here). | — |
