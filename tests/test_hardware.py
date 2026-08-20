"""Red-phase tests for the hardware layer (``ptqnn.hardware``): gate sets, transpilation,
routing, OpenQASM 3 export, executors and the cost model.

Everything a compiler does is checked against the *unitary*, never against a marginal
(``CLAUDE.md``: "test channels, not marginals").  The equivalence used throughout is equality
**modulo a global phase** — a submitted circuit is measured in the computational basis, so that
is exactly the invariance hardware has.  Random angles come from a seeded generator so a failure
is reproducible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ptqnn.backends import reference
from ptqnn.circuits import ir, neurons
from ptqnn.hardware import executor as ex
from ptqnn.hardware import gatesets as gs
from ptqnn.hardware import qasm as qa
from ptqnn.hardware import transpile as tp
from ptqnn.noise.presets import DeviceSpec

ALL_GATESETS = [gs.GATESETS[k] for k in sorted(gs.GATESETS)]
REPORT = Path(__file__).resolve().parents[1] / "reports" / "hardware-compile-table.md"


def unitary(circuit: ir.Circuit) -> np.ndarray:
    """Full unitary of a circuit, column by column, via the reference statevector backend."""
    n = circuit.n_qubits
    eye = np.eye(2**n, dtype=np.complex128)
    return np.array([reference.run_statevector(circuit, eye[i]) for i in range(2**n)]).T


def wrap(ops: list[ir.Gate], n: int) -> ir.Circuit:
    return ir.Circuit(n).extend(ops)


def random_su2(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    return q[0] * np.eye(2, dtype=np.complex128) + 1j * (q[1] * x + q[2] * y + q[3] * z)


def native_names(gset: gs.GateSet) -> set[str]:
    return set(gset.one_qubit) | set(gset.two_qubit)


# --- gate sets --------------------------------------------------------------------------------


def test_every_gateset_cites_a_vendor_url() -> None:
    # Ground-truth discipline: a gate set with no source is a guess (CLAUDE.md).
    for gset in ALL_GATESETS:
        if gset.name == "generic":
            assert "not a vendor" in gset.notes
            continue
        assert "http" in gset.notes, gset.name
        assert gset.one_qubit and gset.two_qubit, gset.name


def test_gateset_lookup_rejects_unknown_names() -> None:
    with pytest.raises(KeyError, match="unknown gate set"):
        gs.gateset("ibm_condor")


# --- (a) CRX ----------------------------------------------------------------------------------


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_decompose_crx_reproduces_ir_crx_for_random_angles(order: tuple[int, int]) -> None:
    # CRX(θ) = (I⊗RZ(-π/2))·CX·(I⊗RY(-θ/2))·CX·(I⊗RY(θ/2))·(I⊗RZ(π/2)); exact, both orders.
    rng = np.random.default_rng(20260820)
    c, t = order
    for theta in [*rng.uniform(-4 * np.pi, 4 * np.pi, 12), 0.0, np.pi, -np.pi]:
        got = unitary(wrap(tp.decompose_crx(c, t, float(theta)), 2))
        want = unitary(ir.Circuit(2).append(ir.crx(c, t, float(theta))))
        assert tp.equal_up_to_phase(got, want), (order, theta)
        # The identity is in fact exact, not merely projective — assert the stronger statement.
        np.testing.assert_allclose(got, want, atol=1e-12)


def test_decompose_crx_costs_exactly_two_cx() -> None:
    ops = tp.decompose_crx(0, 1, 0.3)
    assert sum(1 for g in ops if g.name == "cx") == 2
    assert all(g.n_qubits <= 2 for g in ops)


# --- (b) gate-set rewriting -------------------------------------------------------------------


@pytest.mark.parametrize("gset", ALL_GATESETS, ids=lambda g: g.name)
def test_single_qubit_synthesis_matches_random_unitaries(gset: gs.GateSet) -> None:
    rng = np.random.default_rng(11)
    for _ in range(25):
        u = random_su2(rng)
        ops = tp._synth_1q(u, 0, gset)
        assert {g.name for g in ops} <= native_names(gset), gset.name
        got = unitary(wrap(ops, 1)) if ops else np.eye(2, dtype=np.complex128)
        assert tp.equal_up_to_phase(got, u), gset.name


@pytest.mark.parametrize("gset", ALL_GATESETS, ids=lambda g: g.name)
def test_single_qubit_synthesis_handles_degenerate_cases(gset: gs.GateSet) -> None:
    # Identity, a pure Z rotation and X: the three shapes where the Euler word collapses.
    cases = [
        np.eye(2, dtype=np.complex128),
        ir.rz(0, 0.9).matrix,
        ir.x(0).matrix,
        ir.h(0).matrix,
    ]
    for u in cases:
        ops = tp._synth_1q(u, 0, gset)
        assert {g.name for g in ops} <= native_names(gset)
        got = unitary(wrap(ops, 1)) if ops else np.eye(2, dtype=np.complex128)
        assert tp.equal_up_to_phase(got, u), (gset.name, ops)
    assert tp._synth_1q(np.eye(2, dtype=np.complex128), 0, gset) == []


@pytest.mark.parametrize("gset", ALL_GATESETS, ids=lambda g: g.name)
@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_cx_synthesis_in_each_entangler(gset: gs.GateSet, order: tuple[int, int]) -> None:
    c, t = order
    ops = tp._cx_in_gateset(c, t, gset)
    got = unitary(wrap(ops, 2))
    want = unitary(ir.Circuit(2).append(ir.cx(c, t)))
    assert tp.equal_up_to_phase(got, want), gset.name
    two_q = [g for g in ops if g.n_qubits == 2]
    assert len(two_q) == 1, "one native entangler per CX"
    assert two_q[0].name == gset.entangler


@pytest.mark.parametrize("gset", ALL_GATESETS, ids=lambda g: g.name)
@pytest.mark.parametrize("hidden", ["prototype", "classic"])
def test_to_gateset_preserves_the_unitary_of_a_network(gset: gs.GateSet, hidden: str) -> None:
    rng = np.random.default_rng(4)
    net = neurons.Network(2, 2, hidden=hidden)  # type: ignore[arg-type]
    circ = net.circuit(net.random_params(rng))
    out = tp.to_gateset(circ, gset)
    assert {op.name for op in out.ops} <= native_names(gset), gset.name
    assert tp.equal_up_to_phase(unitary(out), unitary(circ)), gset.name


@pytest.mark.parametrize("gset", ALL_GATESETS, ids=lambda g: g.name)
def test_to_gateset_is_idempotent_on_its_own_output(gset: gs.GateSet) -> None:
    rng = np.random.default_rng(5)
    net = neurons.Network(2, 1)
    circ = net.circuit(net.random_params(rng))
    once = tp.to_gateset(circ, gset)
    twice = tp.to_gateset(once, gset)
    assert tp.equal_up_to_phase(unitary(twice), unitary(circ))
    assert {op.name for op in twice.ops} <= native_names(gset)


def test_to_gateset_rejects_channels() -> None:
    from ptqnn.noise import channels as ch

    circ = ir.Circuit(1)
    circ.append(ir.Channel((0,), list(ch.depolarizing_from_fidelity(0.99, 1)), "depol"))
    with pytest.raises(ValueError, match="channel"):
        tp.to_gateset(circ, gs.GENERIC)


def test_swap_and_cz_have_expansion_rules() -> None:
    for g, n in ((tp.swap(0, 1), 2), (tp.cz(0, 1), 2)):
        circ = ir.Circuit(n).append(g)
        out = tp.to_gateset(circ, gs.GENERIC)
        assert tp.equal_up_to_phase(unitary(out), unitary(circ)), g.name


# --- (c) coupling maps and routing ------------------------------------------------------------


def test_coupling_builders_have_the_expected_shape() -> None:
    assert tp.linear(4) == [(0, 1), (1, 2), (2, 3)]
    assert tp.ring(4) == [(0, 1), (0, 3), (1, 2), (2, 3)]
    assert tp.grid(2, 3) == [(0, 1), (0, 3), (1, 2), (1, 4), (2, 5), (3, 4), (4, 5)]
    assert len(tp.all_to_all(5)) == 10
    # Heavy hex: degree ≤ 3, and BFS relabelling keeps every prefix connected.
    for n in range(2, 28):
        edges = tp.heavy_hex_patch(n)
        deg: dict[int, int] = {}
        for a, b in edges:
            deg[a] = deg.get(a, 0) + 1
            deg[b] = deg.get(b, 0) + 1
        assert max(deg.values()) <= 3, n
        assert set(deg) == set(range(n)), n  # connected ⇒ every node has an edge
        tp._shortest_path(tp._adjacency(edges, n), 0, n - 1)
    with pytest.raises(ValueError, match="27"):
        tp.heavy_hex_patch(28)


@pytest.mark.parametrize("coupling_name", ["linear", "ring", "heavy_hex_patch", "all_to_all"])
def test_routing_preserves_the_unitary_up_to_the_qubit_permutation(coupling_name: str) -> None:
    # A routed circuit computes Q_final · U_ideal · Q_initial†, with Q the basis permutation
    # induced by the logical→physical map.  Anything else means a SWAP was mis-applied.
    n = 5
    rng = np.random.default_rng(6)
    net = neurons.Network(2, 2)
    circ = net.circuit(net.random_params(rng))
    assert circ.n_qubits == n
    coupling = getattr(tp, coupling_name)(n)
    routed, final_map = tp.route(circ, coupling)
    q0 = tp.permutation_unitary({q: q for q in range(n)}, n)
    qf = tp.permutation_unitary(final_map, n)
    assert tp.equal_up_to_phase(unitary(routed), qf @ unitary(circ) @ q0.conj().T)


def test_routing_honours_a_non_trivial_initial_layout() -> None:
    n = 5
    rng = np.random.default_rng(7)
    net = neurons.Network(2, 2)
    circ = net.circuit(net.random_params(rng))
    layout = [4, 0, 2, 1, 3]
    routed, final_map = tp.route(circ, tp.linear(n), initial_layout=layout)
    q0 = tp.permutation_unitary({q: p for q, p in enumerate(layout)}, n)
    qf = tp.permutation_unitary(final_map, n)
    assert tp.equal_up_to_phase(unitary(routed), qf @ unitary(circ) @ q0.conj().T)


def test_routed_circuit_only_uses_coupled_pairs() -> None:
    rng = np.random.default_rng(8)
    net = neurons.Network(3, 2)  # 6 qubits, star-like: every input touches every hidden qubit
    circ = net.circuit(net.random_params(rng))
    coupling = tp.linear(6)
    routed, _ = tp.route(circ, coupling)
    allowed = set(coupling)
    for op in routed.ops:
        if len(op.qubits) == 2:
            a, b = op.qubits
            assert (min(a, b), max(a, b)) in allowed, op
    assert routed.count("swap") > 0, "a linear chain cannot host a star without SWAPs"


def test_routing_on_all_to_all_inserts_no_swaps() -> None:
    rng = np.random.default_rng(9)
    net = neurons.Network(2, 4)
    circ = net.circuit(net.random_params(rng))
    routed, final_map = tp.route(circ, tp.all_to_all(circ.n_qubits))
    assert routed.count("swap") == 0
    assert final_map == {q: q for q in range(circ.n_qubits)}


def test_routing_rejects_a_bad_layout() -> None:
    circ = ir.Circuit(3).append(ir.cx(0, 2))
    with pytest.raises(ValueError, match="injective"):
        tp.route(circ, tp.linear(3), initial_layout=[0, 1, 1])
    with pytest.raises(ValueError, match="every logical qubit"):
        tp.route(circ, tp.linear(3), initial_layout=[0, 1])


def test_permutation_unitary_is_a_permutation() -> None:
    m = {0: 2, 1: 0, 2: 1}
    q = tp.permutation_unitary(m, 3)
    np.testing.assert_allclose(q.conj().T @ q, np.eye(8))
    # |100⟩ (qubit 0 set) must land on the basis state with qubit 2 set, i.e. |001⟩.
    v = np.zeros(8, dtype=np.complex128)
    v[0b100] = 1
    assert np.argmax(np.abs(q @ v)) == 0b001


# --- (d) metrics ------------------------------------------------------------------------------


def test_metrics_counts_depth_gates_and_critical_path() -> None:
    circ = ir.Circuit(3)
    circ.extend([ir.h(0), ir.h(1), ir.cx(0, 1), tp.swap(1, 2), ir.rz(2, 0.4), ir.cx(0, 2)])
    m = tp.metrics(circ)
    assert m["n_1q"] == 3
    assert m["n_2q"] == 3
    assert m["n_swap"] == 1
    assert m["n_qubits"] == 3
    # ASAP layers: [h0,h1] | cx01 | swap12 | rz2 | cx02  →  depth 5.
    assert m["depth"] == 5
    assert m["critical_path_ns"] is None

    timed = tp.metrics(circ, {"1q": 10.0, "2q": 100.0})
    # h0 (10) → cx01 (100) → swap12 (3×100) → rz2 (10) → cx02 (100) = 520 ns on the busy wire.
    assert timed["critical_path_ns"] == pytest.approx(520.0)
    explicit = tp.metrics(circ, {"1q": 10.0, "2q": 100.0, "swap": 50.0})
    assert explicit["critical_path_ns"] == pytest.approx(270.0)


def test_metrics_of_an_empty_circuit() -> None:
    m = tp.metrics(ir.Circuit(2))
    assert (m["depth"], m["n_1q"], m["n_2q"], m["n_swap"]) == (0, 0, 0, 0)


# --- OpenQASM 3 -------------------------------------------------------------------------------


def test_openqasm3_declares_qubits_measurements_and_bit_order() -> None:
    circ = ir.Circuit(2).extend([ir.h(0), ir.cx(0, 1)])
    src = qa.to_openqasm3(circ)
    assert src.startswith("OPENQASM 3.0;")
    assert 'include "stdgates.inc";' in src
    assert "qubit[2] q;" in src and "bit[2] c;" in src
    assert "c[0] = measure q[0];" in src and "c[1] = measure q[1];" in src
    assert "msb-first" in qa.BITSTRING_ORDER
    # Our qubit 0 is the leftmost bit; a QASM/Aer key is the reverse string.
    assert qa.from_qasm_bitstring("011") == "110"
    assert qa.counts_from_qasm({"01": 3, "10": 5}) == {"10": 3, "01": 5}


def test_openqasm3_refuses_channels_and_initial_states() -> None:
    from ptqnn.noise import channels as ch

    noisy = ir.Circuit(1)
    noisy.append(ir.Channel((0,), list(ch.depolarizing_from_fidelity(0.99, 1)), "depol"))
    with pytest.raises(ValueError, match="channel"):
        qa.to_openqasm3(noisy)

    psi = np.array([1, 1, 0, 0], dtype=np.complex128) / np.sqrt(2)
    with pytest.raises(ValueError, match="initial_state"):
        qa.to_openqasm3(ir.Circuit(2, initial_state=psi))


def test_openqasm3_round_trips_through_qiskit() -> None:
    qiskit = pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_qasm3_import")
    from qiskit.quantum_info import Operator

    circ = ir.Circuit(3)
    circ.extend(
        [
            ir.h(0),
            ir.rx(1, 0.41),
            ir.ry(2, -0.93),
            ir.rz(0, 1.17),
            ir.cx(0, 1),
            tp.cz(1, 2),
            ir.crx(2, 0, 0.77),
            ir.x(1),
            tp.swap(0, 2),
            tp.sx(1),
        ]
    )
    loaded = qiskit.qasm3.loads(qa.to_openqasm3(circ, measure=False))
    # Qiskit's q[0] is the least significant bit, ours is the most significant, so the same
    # OpenQASM text names the same gates on the same wires but a bit-reversed basis index.
    r = qa.bit_reversal_permutation(3)
    got = r @ Operator(loaded).data @ r.conj().T
    assert tp.equal_up_to_phase(got, unitary(circ))


@pytest.mark.parametrize("gset", ALL_GATESETS, ids=lambda g: g.name)
def test_openqasm3_round_trips_native_gate_definitions(gset: gs.GateSet) -> None:
    qiskit = pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_qasm3_import")
    from qiskit.quantum_info import Operator

    rng = np.random.default_rng(12)
    net = neurons.Network(2, 1)
    circ = tp.to_gateset(net.circuit(net.random_params(rng)), gset)
    loaded = qiskit.qasm3.loads(qa.to_openqasm3(circ, measure=False))
    r = qa.bit_reversal_permutation(circ.n_qubits)
    got = r @ Operator(loaded).data @ r.conj().T
    assert tp.equal_up_to_phase(got, unitary(circ)), gset.name


def test_openqasm3_rejects_unknown_gate_names() -> None:
    weird = ir.unitary((0,), np.eye(2), name="frobnicate")
    with pytest.raises(NotImplementedError, match="frobnicate"):
        qa.to_openqasm3(ir.Circuit(1).append(weird))


# --- executors --------------------------------------------------------------------------------


def sample_circuit() -> ir.Circuit:
    circ = ir.Circuit(3)
    circ.extend(
        [ir.h(0), ir.rx(1, 0.7), ir.crx(0, 2, 1.3), ir.cx(1, 2), ir.ry(0, -0.4), ir.crx(2, 1, 0.9)]
    )
    return circ


def test_local_executor_counts_use_our_bit_order() -> None:
    circ = ir.Circuit(2).append(ir.x(0))  # qubit 0 set ⇒ the key must be "10"
    counts = ex.LocalExecutor(reference.StatevectorBackend(), seed=3).run(circ, 64)
    assert counts == {"10": 64}
    assert ex.LocalExecutor(reference.StatevectorBackend()).describe()["kind"] == "local"


def test_local_executor_is_seed_deterministic() -> None:
    circ = sample_circuit()
    a = ex.LocalExecutor(reference.StatevectorBackend(), seed=1).run(circ, 500)
    b = ex.LocalExecutor(reference.StatevectorBackend(), seed=1).run(circ, 500)
    assert a == b
    assert sum(a.values()) == 500


def test_to_qiskit_reverses_the_qubit_order_and_keeps_the_unitary() -> None:
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Operator

    circ = sample_circuit()
    # With our qubit q on Qiskit wire n-1-q the two significance conventions coincide, so the
    # Qiskit operator equals the IR unitary with no permutation at all.
    assert tp.equal_up_to_phase(Operator(ex.to_qiskit(circ)).data, unitary(circ))


def test_aer_and_local_executors_agree_in_distribution() -> None:
    pytest.importorskip("qiskit_aer")
    circ = sample_circuit()
    shots = 20_000
    local = ex.LocalExecutor(reference.StatevectorBackend(), seed=2024).run(circ, shots)
    aer = ex.QiskitAerExecutor(seed=2024).run(circ, shots)
    assert set(aer) <= {format(i, "03b") for i in range(8)}
    p_local = ex.counts_to_probabilities(local, 3)
    p_aer = ex.counts_to_probabilities(aer, 3)
    p_exact = reference.StatevectorBackend().probabilities(circ)
    # Two independent multinomials of 20 000 shots over 8 outcomes: the TV distance between each
    # and the exact distribution is ~sqrt(8/(2π·20000)) ≈ 0.008, so 0.03 is a ~4σ bound.
    assert ex.total_variation(p_local, p_exact) < 0.03
    assert ex.total_variation(p_aer, p_exact) < 0.03
    assert ex.total_variation(p_local, p_aer) < 0.03


def test_aer_executor_reproduces_a_deterministic_bit_string() -> None:
    pytest.importorskip("qiskit_aer")
    circ = ir.Circuit(3).extend([ir.x(0), ir.x(2)])  # qubits 0 and 2 set ⇒ "101"
    assert ex.QiskitAerExecutor(seed=1).run(circ, 128) == {"101": 128}


def test_aer_executor_accepts_a_noise_model() -> None:
    pytest.importorskip("qiskit_aer")
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(0.05, 1), ["x"])
    circ = ir.Circuit(1).append(ir.x(0))
    counts = ex.QiskitAerExecutor(noise_model=nm, seed=5).run(circ, 4000)
    assert counts.get("0", 0) > 0, "5 % depolarising must leak some |0⟩ outcomes"
    assert ex.QiskitAerExecutor(noise_model=nm).describe()["noisy"] is True


def test_ibm_runtime_dry_run_builds_an_isa_circuit_without_a_network_call() -> None:
    pytest.importorskip("qiskit")
    rng = np.random.default_rng(13)
    net = neurons.Network(2, 4)  # Flappy Bird size: 7 qubits [PTQNN Fig 10]
    circ = net.circuit(net.random_params(rng))
    exe = ex.IBMRuntimeExecutor(backend_name="ibm_heron")
    payload = exe.dry_run_payload(circ, shots=4096)

    assert payload["would_submit"]["primitive"] == "SamplerV2"
    assert payload["would_submit"]["shots"] == 4096
    assert payload["network"].startswith("none")
    isa = payload["isa"]
    assert set(isa["ops"]) <= set(ex.IBM_HERON_BASIS) | {"barrier", "measure"}
    assert isa["n_2q"] > 0 and isa["depth"] > 0
    assert isa["qasm3"].startswith("OPENQASM 3.0;")
    assert exe.describe()["installed"] is False

    with pytest.raises(NotImplementedError, match="qiskit-ibm-runtime"):
        exe.run(circ, 100)


@pytest.mark.parametrize(
    ("factory", "needle"),
    [(ex.BraketExecutor, "amazon-braket-sdk"), (ex.QuantinuumExecutor, "pytket-quantinuum")],
)
def test_stub_executors_name_what_they_need(factory: type, needle: str) -> None:
    exe = factory()
    with pytest.raises(NotImplementedError, match=needle):
        exe.run(ir.Circuit(1), 10)
    assert exe.describe()["requires"]


def test_executors_satisfy_the_protocol() -> None:
    for exe in (
        ex.LocalExecutor(reference.StatevectorBackend()),
        ex.QiskitAerExecutor(),
        ex.IBMRuntimeExecutor(),
        ex.BraketExecutor(),
        ex.QuantinuumExecutor(),
    ):
        assert isinstance(exe, ex.Executor)


# --- compile_for and the cost model -----------------------------------------------------------


def spec(
    name: str, vendor: str, one: tuple[str, ...], two: tuple[str, ...], topo: str
) -> DeviceSpec:
    """A minimal DeviceSpec: only the fields compile_for reads.  No invented calibration."""
    return DeviceSpec(
        name=name,
        vendor=vendor,
        technology="test fixture",
        n_qubits=27,
        topology=topo,
        native_1q=one,
        native_2q=two,
        t1_us=None,
        t2_us=None,
        gate_1q_error=0.0,
        gate_1q_ns=0.0,
        gate_2q_error=0.0,
        gate_2q_ns=0.0,
        readout_p01=0.0,
        readout_p10=0.0,
        readout_ns=0.0,
    )


FIXTURES = [
    spec("ibm_heron_like", "IBM", ("rz", "sx", "x"), ("cz",), "heavy-hex"),
    spec("iqm_like", "IQM", ("prx",), ("cz",), "grid"),
    spec("rigetti_like", "Rigetti", ("rx", "rz"), ("cz", "iswap"), "grid"),
    spec("quantinuum_like", "Quantinuum", ("U1q", "Rz"), ("ZZ", "RZZ"), "all-to-all"),
    spec("ionq_like", "IonQ", ("gpi", "gpi2"), ("ms",), "all-to-all"),
    spec("generic_like", "ACME", ("rz", "ry", "rx"), ("cx",), "linear"),
]


@pytest.mark.parametrize("device", FIXTURES, ids=lambda d: d.name)
def test_gateset_and_coupling_are_chosen_from_the_device_spec(device: DeviceSpec) -> None:
    chosen = tp.gateset_for(device.native_2q, device.native_1q)
    expected = {
        "ibm_heron_like": "ibm_heron",
        "iqm_like": "iqm",
        "rigetti_like": "rigetti_ankaa",
        "quantinuum_like": "quantinuum_h",
        "ionq_like": "ionq",
        "generic_like": "generic",
    }[device.name]
    assert chosen.name == expected
    coupling, note = tp.coupling_for(device.topology, 7)
    assert coupling and isinstance(note, str)


def test_compile_for_flappy_network_is_native_and_faithful() -> None:
    rng = np.random.default_rng(14)
    net = neurons.Network(2, 4)  # [PTQNN Fig 10]: 2 inputs, 4 prototypes, 1 output = 7 qubits
    circ = net.circuit(net.random_params(rng))
    assert circ.n_qubits == 7
    for device in FIXTURES:
        compiled = tp.compile_for(circ, device)
        allowed = native_names(compiled.gateset)
        assert {op.name for op in compiled.circuit.ops} <= allowed, device.name
        q0 = tp.permutation_unitary({q: q for q in range(7)}, 7)
        qf = tp.permutation_unitary(compiled.qubit_map, 7)
        assert tp.equal_up_to_phase(unitary(compiled.circuit), qf @ unitary(circ) @ q0.conj().T), (
            device.name
        )
        assert compiled.metrics["n_2q"] > 0
        assert compiled.notes


def test_compile_for_refuses_a_circuit_bigger_than_the_device() -> None:
    small = spec("tiny", "ACME", ("rz", "ry", "rx"), ("cx",), "linear")
    tiny = DeviceSpec(**{**small.__dict__, "n_qubits": 3})
    with pytest.raises(ValueError, match="needs 5 qubits"):
        tp.compile_for(ir.Circuit(5), tiny)


def test_quantinuum_hqc_matches_the_published_formula() -> None:
    rng = np.random.default_rng(15)
    net = neurons.Network(2, 4)
    compiled = tp.compile_for(net.circuit(net.random_params(rng)), FIXTURES[3])
    est = ex.cost_estimate(compiled, shots=1000, device=FIXTURES[3])
    n1 = int(compiled.metrics["n_1q"] or 0)
    n2 = int(compiled.metrics["n_2q"] or 0)
    n_m = 2 * 7
    assert est.unit == "HQC"
    assert est.value == pytest.approx(5.0 + (n1 + 10 * n2 + 5 * n_m) / 5000.0 * 1000)
    assert any("docs.quantinuum.com" in s for s in est.sources)


def test_cost_estimates_for_the_other_vendors() -> None:
    rng = np.random.default_rng(16)
    net = neurons.Network(2, 4)
    circ = net.circuit(net.random_params(rng))

    ionq = ex.cost_estimate(tp.compile_for(circ, FIXTURES[4]), 1000, FIXTURES[4])
    assert ionq.unit == "USD" and ionq.value > 12.4
    assert any("[ASSUMPTION]" in a for a in ionq.assumptions)

    # Braket dispatch needs both: an AWS/Braket vendor string and a device slug in the price
    # table (the same silicon priced directly by IonQ follows IonQ's own formula instead).
    braket_device = spec("ionq_forte", "AWS Braket", ("gpi", "gpi2"), ("zz",), "all-to-all")
    braket = ex.cost_estimate(tp.compile_for(circ, braket_device), 1000, braket_device)
    assert braket.model == "aws_braket"
    assert braket.value == pytest.approx(0.30 + 0.08 * 1000)

    ibm = ex.cost_estimate(tp.compile_for(circ, FIXTURES[0]), 1000, FIXTURES[0])
    assert ibm.model == "ibm_pay_as_you_go"
    assert any("UNVERIFIED" in a for a in ibm.assumptions)

    generic = ex.cost_estimate(tp.compile_for(circ, FIXTURES[5]), 1000, FIXTURES[5])
    assert generic.unit == "seconds"
    assert generic.value == pytest.approx(30.0)  # no durations published ⇒ overhead only


def test_compile_table_for_flappy_and_bitcount(capsys: pytest.CaptureFixture[str]) -> None:
    """Depth / 2q count / SWAP count for every gate set × topology, on the two sizes that matter.

    Flappy Bird is ``Network(2, 4)`` = 7 qubits ``[PTQNN Fig 10]``; the bit-count task of
    ``[PTQNN Fig 5]`` is ``Network(5, 5)`` = 11 qubits.  The table is written to
    ``reports/hardware-compile-table.md`` and printed with ``-s``.
    """
    rng = np.random.default_rng(17)
    circuits = {
        "flappy Network(2,4), 7q": neurons.Network(2, 4),
        "bit-count Network(5,5), 11q": neurons.Network(5, 5),
    }
    topologies = ["linear", "heavy-hex", "all-to-all"]

    lines = [
        "# Transpiled circuit cost: gate set × topology",
        "",
        "Generated by `tests/test_hardware.py::test_compile_table_for_flappy_and_bitcount`.",
        "",
        "`depth` is the ASAP layer count of the *native* circuit; `n_2q` counts every native",
        "entangler in it, including the three each routed SWAP costs; `n_swap` is the number of",
        "SWAPs the router inserted (each CRX costs 2 entanglers, each SWAP 3, so",
        "`n_2q = 2·n_CRX + 3·n_swap`).  Routing is greedy",
        "nearest-neighbour and **not optimal** — see `ptqnn.hardware.transpile.route`.",
        "The prototype network is star-like (every input couples to every hidden qubit), which",
        "is why a linear chain costs so many SWAPs and an all-to-all trapped-ion machine none.",
        "",
    ]

    for label, net in circuits.items():
        circ = net.circuit(net.random_params(rng))
        base = tp.metrics(circ)
        lines += [
            f"## {label}",
            "",
            f"Logical circuit: depth {base['depth']}, {base['n_2q']} CRX, {base['n_1q']} 1q gates.",
            "",
            "| gate set | topology | depth | n_1q | n_2q | n_swap |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for gset in ALL_GATESETS:
            for topo in topologies:
                device = spec(
                    f"{gset.name}_{topo}",
                    "fixture",
                    gset.one_qubit,
                    gset.two_qubit,
                    topo,
                )
                # Force the gate set (the fixture's native names already select it).
                assert tp.gateset_for(gset.two_qubit, gset.one_qubit).name == gset.name
                compiled = tp.compile_for(circ, device)
                m = compiled.metrics
                assert {op.name for op in compiled.circuit.ops} <= native_names(gset)
                # Each routed SWAP became three native entanglers in the final basis.
                assert int(m["n_2q"] or 0) >= int(m["n_2q_from_swap"] or 0)
                assert int(m["n_2q_from_swap"] or 0) == 3 * int(m["n_swap"] or 0)
                lines.append(
                    f"| {gset.name} | {topo} | {m['depth']} | {m['n_1q']} | "
                    f"{m['n_2q']} | {m['n_swap']} |"
                )
        lines.append("")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    assert REPORT.exists()
    captured = capsys.readouterr()
    assert "n_swap" in captured.out
