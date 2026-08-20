"""Cross-check the reference backend against the independently written oracle simulator.

``tests/oracle/indep_sim.py`` was produced blind (Codex, brief ``notes/briefs/codex-01``) from
the same ordering convention and shares no code with :mod:`ptqnn`.  Agreement on full density
matrices — not marginals — over random circuits with CRX, RX, H and Kraus channels is the
primary evidence that the tensor-reshape machinery in ``reference.py`` is right.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ptqnn.backends import reference as ref
from ptqnn.circuits import ir, neurons
from tests.oracle import indep_sim as oracle


def _random_kraus_1q(rng: np.random.Generator) -> list[np.ndarray]:
    # Random 1-qubit channel from a Haar-ish 4x2 isometry V: K_i = V[2i:2i+2, :].
    a = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
    v, _ = np.linalg.qr(a)
    return [v[0:2, :], v[2:4, :]]


def _random_circuit(n: int, rng: np.random.Generator, depth: int) -> ir.Circuit:
    circ = ir.Circuit(n)
    for _ in range(depth):
        r = rng.random()
        q = int(rng.integers(n))
        if r < 0.3:
            circ.append(ir.rx(q, float(rng.uniform(-np.pi, np.pi))))
        elif r < 0.45:
            circ.append(ir.h(q))
        elif r < 0.8 and n > 1:
            c, t = (int(v) for v in rng.choice(n, 2, replace=False))
            circ.append(ir.crx(c, t, float(rng.uniform(-np.pi, np.pi))))
        else:
            circ.append(ir.Channel((q,), _random_kraus_1q(rng), "rand"))
    return circ


def _to_oracle_ops(circ: ir.Circuit) -> list[tuple[object, ...]]:
    ops: list[tuple[object, ...]] = []
    for op in circ.ops:
        if isinstance(op, ir.Gate):
            if op.name == "crx":
                ops.append(("crx", op.qubits[0], op.qubits[1], op.params[0]))
            elif op.name == "rx":
                ops.append(("rx", op.qubits[0], op.params[0]))
            elif op.name == "h":
                ops.append(("h", op.qubits[0]))
            else:
                ops.append(("u", list(op.qubits), op.matrix))
        else:
            ops.append(("kraus", list(op.qubits), list(op.kraus)))
    return ops


@settings(max_examples=30, deadline=None)
@given(st.integers(min_value=1, max_value=5), st.integers(min_value=0, max_value=2**31 - 1))
def test_reference_matches_oracle_on_random_noisy_circuits(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    circ = _random_circuit(n, rng, depth=10)
    got = ref.run_density_matrix(circ)
    expected = oracle.run(n, _to_oracle_ops(circ))
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_two_qubit_channel_matches_oracle() -> None:
    n = 4
    rng = np.random.default_rng(5)
    a = rng.normal(size=(8, 4)) + 1j * rng.normal(size=(8, 4))
    v, _ = np.linalg.qr(a)
    ks = [v[0:4, :], v[4:8, :]]
    for qubits in [(1, 3), (3, 1), (0, 2)]:
        circ = ir.Circuit(n, [ir.h(q) for q in range(n)])
        circ.append(ir.Channel(qubits, ks, "rand2"))
        got = ref.run_density_matrix(circ)
        expected = oracle.run(n, _to_oracle_ops(circ))
        np.testing.assert_allclose(got, expected, atol=1e-12)


def test_prototype_network_matches_oracle() -> None:
    net = neurons.Network(n_inputs=3, n_hidden=2, hidden="prototype")
    params = net.random_params(np.random.default_rng(11))
    circ = net.circuit(params, prep=[ir.h(q) for q in net.input_qubits])
    psi = ref.run_statevector(circ)
    expected = oracle.run(net.n_qubits, _to_oracle_ops(circ))
    np.testing.assert_allclose(np.outer(psi, psi.conj()), expected, atol=1e-12)
    assert np.trace(expected).real == pytest.approx(1.0)
