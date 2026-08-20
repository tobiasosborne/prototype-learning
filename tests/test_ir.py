"""Red-phase tests for the circuit IR (``ptqnn.circuits.ir``).

Convention under test: qubit 0 is the most significant bit of the basis index
(``index = sum(b_q << (n - 1 - q))``).  Every op is a CP map; unitary ops carry a witness.
"""

import numpy as np
import pytest

from ptqnn.circuits import ir


def test_rx_matrix_matches_definition() -> None:
    # RX(θ) = exp(-iθX/2) = [[cos θ/2, -i sin θ/2], [-i sin θ/2, cos θ/2]]  [SNN §III.A convention]
    th = 0.7
    g = ir.rx(0, th)
    c, s = np.cos(th / 2), np.sin(th / 2)
    expected = np.array([[c, -1j * s], [-1j * s, c]])
    np.testing.assert_allclose(g.matrix, expected, atol=1e-15)
    assert g.qubits == (0,)
    assert g.name == "rx"
    assert g.params == (th,)


def test_rx_on_zero_gives_sin_squared_activation() -> None:
    # P(1) = sin²(θ/2) for RX(θ)|0⟩: the stochastic neuron's activation law [SNN Fig 2 / §III].
    th = 1.1
    psi = ir.rx(0, th).matrix @ np.array([1, 0])
    assert abs(psi[1]) ** 2 == pytest.approx(np.sin(th / 2) ** 2)


def test_crx_block_structure_control_first() -> None:
    # CRX(θ) with control = first listed qubit: |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ RX(θ)
    th = 0.3
    g = ir.crx(2, 5, th)
    assert g.qubits == (2, 5)
    expected = np.eye(4, dtype=complex)
    expected[2:, 2:] = ir.rx(0, th).matrix
    np.testing.assert_allclose(g.matrix, expected, atol=1e-15)


def test_unitary_witness_rejects_nonunitary() -> None:
    with pytest.raises(ValueError, match="unitary"):
        ir.Gate(qubits=(0,), matrix=np.array([[1, 1], [0, 1]], dtype=complex), name="bad")


def test_kraus_witness_rejects_non_trace_preserving() -> None:
    with pytest.raises(ValueError, match="trace"):
        ir.Channel(qubits=(0,), kraus=[np.eye(2) * 0.5], name="bad")


def test_kraus_accepts_amplitude_damping() -> None:
    gamma = 0.2
    k0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
    k1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
    ch = ir.Channel(qubits=(1,), kraus=[k0, k1], name="amp_damp")
    assert ch.n_qubits == 1
    assert len(ch.kraus) == 2


def test_gate_as_channel_is_single_kraus() -> None:
    g = ir.h(0)
    ch = g.as_channel()
    assert len(ch.kraus) == 1
    np.testing.assert_allclose(ch.kraus[0], g.matrix)


def test_circuit_tracks_qubit_bounds() -> None:
    c = ir.Circuit(3)
    c.append(ir.h(0))
    c.append(ir.crx(0, 2, 0.1))
    assert len(c.ops) == 2
    with pytest.raises(ValueError, match="qubit"):
        c.append(ir.rx(3, 0.1))
    with pytest.raises(ValueError, match="distinct"):
        ir.crx(1, 1, 0.2)


def test_circuit_gate_counts() -> None:
    c = ir.Circuit(3, [ir.h(0), ir.h(1), ir.crx(0, 2, 0.1), ir.crx(1, 2, 0.1), ir.rx(2, 0.5)])
    assert c.count("crx") == 2
    assert c.count("h") == 2
    assert c.n_two_qubit_gates() == 2


def test_dagger_inverts() -> None:
    g = ir.crx(0, 1, 0.9)
    np.testing.assert_allclose(g.dagger().matrix @ g.matrix, np.eye(4), atol=1e-14)
    assert ir.rx(0, 0.4).dagger().params == (-0.4,)
