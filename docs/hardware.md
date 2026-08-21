# Hardware layer

`ptqnn/hardware/` rewrites the paper's circuits into vendor native gate sets on vendor
topologies, exports OpenQASM 3, wraps execution behind one `Executor` protocol (dry-run only —
no credentials exist in this repo), and prices shots. `experiments/hardware_plan.py` turns all of
it into the numbers `reports/hardware-plan.md` quotes. Nothing has been submitted to a QPU.

## Gate sets (`gatesets.py`)

`GateSet(name, one_qubit, two_qubit, notes)` records only what the vendor documents (URL and
quoted fragment in `notes`); numbers live in the device TOMLs. Lower-case IR names throughout:
`IBM_HERON` (`rz, sx, x` / `cz`), `QUANTINUUM_H` (`u1q, rz` / `rzz, zz`), `IONQ` (`gpi, gpi2` /
`ms`), `IQM` (`prx` / `cz`), `RIGETTI_ANKAA` (`rx, rz` / `iswap` — not synthesised, see notes),
`GENERIC` (`rz, ry, rx` / `cx`). `gateset(name)`, `GateSet.entangler` (the first 2q gate the
transpiler can target). Matrix conventions for the non-standard names are in the module docstring
(`prx(θ,φ) = RZ(φ)RX(θ)RZ(−φ)`, `gpi`, `gpi2`, `rzz`, `ms`).

## Transpilation (`transpile.py`)

Three passes, each verified numerically in `tests/test_hardware.py` modulo global phase (the only
equivalence a measured circuit sees; `equal_up_to_phase`):

1. **Expand to {1q, CX}.** The only entangler the paper uses is `CRX(θ)`; no vendor has it.
   `CRX(θ) = (I⊗RZ(−π/2))·CX·(I⊗RY(−θ/2))·CX·(I⊗RY(θ/2))·(I⊗RZ(π/2))` — exact, 2 CX
   (`decompose_crx`).
2. **CX → native entangler.** `CX = (I⊗H)·CZ·(I⊗H)`; `CZ ≐ (RZ(−π/2)⊗RZ(−π/2))·RZZ(π/2)`;
   `XX(θ) = (H⊗H)·RZZ(θ)·(H⊗H)` so `MS = XX(π/2)`. One native entangler per CX for ZZ/RZZ
   (Quantinuum) and MS (IonQ) sets; one CZ plus glue for IBM/IQM/Rigetti.
3. **1q re-synthesis.** Maximal 1q runs per wire are multiplied into one `2×2` and re-emitted
   from ZYZ angles in fixed-form Euler words: generic 3 gates; IBM `RZ·SX·RZ·SX·RZ` (5);
   Quantinuum `RZ·U1q` (2); IQM 3 `PRX`; IonQ `GPI2·GPI2·GPI·GPI` (4). Zero rotations (mod 2π)
   are dropped. Fixed-form, checkable, not claimed optimal.

`to_gateset(circuit, gs)` does all three. Routing (`route(circuit, coupling, initial_layout)`)
is deliberately naive — greedy nearest-neighbour SWAP insertion along BFS shortest paths, no
lookahead, no layout search — so cross-topology comparisons are honest without pretending to be a
compiler; a real submission uses the vendor's. Couplings: `linear`, `ring`, `grid`,
`heavy_hex_patch` (first `n` qubits of IBM's 27-q tile in BFS order), `all_to_all`.
`metrics(circuit, durations_ns)` gives depth, gate counts, SWAPs, critical-path time.
`compile_for(circuit, device, layout)` → `CompiledCircuit(circuit, qubit_map, metrics, gateset,
coupling, device_name, notes)` straight from a `DeviceSpec`; `gateset_note_for`/`coupling_for`
emit an `[ASSUMPTION]` note when a device's advertised gates or topology have to be approximated
(synthetic patches, gate-set fallback — assumption A10 in the report).

Cost per CRX: 2 native entanglers; each routed SWAP 3, so `n_2q = 2·n_CRX + 3·n_swap`. The
prototype network is complete bipartite (every input to every hidden qubit) — the worst fit for a
lattice: bit-count's 30 CRX cost 60 entanglers all-to-all, 327 on heavy-hex, 540 on a line
(`reports/hardware-compile-table.md`).

## OpenQASM 3 (`qasm.py`)

`to_openqasm3(circuit)` is index-preserving (our qubit `q` is `q[q]`, measured into `c[q]`).
What is *not* preserved is significance: OpenQASM/Qiskit print `c[n−1]` leftmost, we print
qubit 0 leftmost (`BITSTRING_ORDER`, the Fig 5 convention). The two differ by a string reversal
and nothing else: `from_qasm_bitstring`, `counts_from_qasm`, `bit_reversal_permutation`.

## Executors (`executor.py`)

`Executor.run(circuit, shots) -> {bitstring: count}` in our bit order; `describe()`.

| class | what it does | status |
|---|---|---|
| `LocalExecutor(backend, seed)` | `reference.sample_counts` on any of our backends | the oracle |
| `QiskitAerExecutor(noise_model, seed, method)` | independent implementation; `to_qiskit` reverses qubit order so Aer keys need no re-ordering | cross-checked vs Local in TVD at 20 000 shots |
| `IBMRuntimeExecutor(backend_name, dry_run, coupling)` | builds the ISA circuit a job would carry (`dry_run_payload`), qiskit O1–O3 | dry run only |
| `BraketExecutor`, `QuantinuumExecutor` | stubs that raise with instructions | not implemented |

`cost_estimate(compiled, shots, device)` → `CostEstimate(model, unit, value, breakdown, sources,
assumptions)`: Quantinuum HQC formula, Azure IonQ per-gate-shot pricing, Braket per-shot + $0.30
per task (prices retrieved 2026-08-20; devices without a current price row are absent, not
guessed), and a wall-clock model `shots × circuit_time + overhead` otherwise.

## The hardware plan pipeline (`experiments/hardware_plan.py`)

Review 002 blocker 1: v1 of the hardware report cited no run dir and a dozen numbers were wrong.
Now every quantitative claim in `reports/hardware-plan.md` is printed by this script into
`experiments/runs/hardware-plan/<stamp>/{numbers.json, numbers.md, meta.json}` and the prose
quotes `numbers.md` only. Sections: `logical` (qubits/depth/CRX/1q per paper network),
`compiled` (per real preset via `compile_for`), `fidelity` (depolarising product over native
gates × relaxation over each qubit's ASAP idle gaps — the full-path variant is printed beside it
as pessimistic, review 003 #5), `sabre` (qiskit O1–O3 on Fez for comparison with our router),
`cost`, `shots` (shot budgets from decision margins), `noise_run` (simulated TVD of *compiled*
circuits — 5–15× the logical ones), `stateprep` (Iris: cost of preparing the superposition over
the 12–14 distinct training codes, not 120 samples — review 003 blocker), `provenance` (every
number joined to a TOML provenance entry by exact field id and value; fails closed).
`ASSUMED_1Q_NS` supplies the 40 ns 1q duration IQM and Rigetti do not publish.

Rules: never hand-edit a number in the report — change the script, re-run (≈ 2 min on GPU),
regenerate; `reports/tex` figures come from `experiments/report_figures.py` the same way.

## What the plan concludes (short)

Flappy (7 q, 12 CRX) and ECG (7–9 q) are runnable today on any preset; bit-count (11 q, 30 CRX)
is marginal on superconducting lattices because of routing and reasonable on trapped ions; Iris
hinges on sparse state preparation and cannot be settled without an actual preparation circuit.
Shot budgets, not gate fidelity, dominate Flappy (decision margins ≈ 0.02). Details, per-device
fidelity products and prices: `reports/hardware-plan.md`, `reports/tex/sections/06-hardware.tex`.
