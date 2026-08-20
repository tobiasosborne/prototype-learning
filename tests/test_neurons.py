"""Red-phase tests for neurons and the network builder (``ptqnn.circuits.neurons``)."""

import numpy as np
import pytest

from ptqnn.backends import reference as ref
from ptqnn.circuits import ir, neurons


def p_out1(circ: ir.Circuit, out: int) -> float:
    p = ref.probabilities(ref.run_statevector(circ))
    return float(ref.marginal(p, [out])[1])


def test_classic_neuron_gate_sequence() -> None:
    # [PTQNN Fig 1]: bias RX(θ) on q3, then CRX(φ1) q1→q3, CRX(φ2) q2→q3.
    ops = neurons.classic_neuron(inputs=(0, 1), out=2, weights=(0.3, 0.4), bias=0.5)
    names = [(o.name, o.qubits, o.params) for o in ops]
    assert names == [
        ("rx", (2,), (0.5,)),
        ("crx", (0, 2), (0.3,)),
        ("crx", (1, 2), (0.4,)),
    ]


def test_prototype_neuron_gate_sequence() -> None:
    # [PTQNN Fig 2]: per input i: RX(φ_i) on q_i, CRX(θ) q_i→q3, RX(−φ_i) on q_i. Same θ for both.
    ops = neurons.prototype_neuron(inputs=(0, 1), out=2, prototype=(0.3, 0.4), theta=1.2)
    names = [(o.name, o.qubits, o.params) for o in ops]
    assert names == [
        ("rx", (0,), (0.3,)),
        ("crx", (0, 2), (1.2,)),
        ("rx", (0,), (-0.3,)),
        ("rx", (1,), (0.4,)),
        ("crx", (1, 2), (1.2,)),
        ("rx", (1,), (-0.4,)),
    ]


def test_classic_neuron_activation_law_on_basis_input() -> None:
    # For basis input b and classic neuron: out qubit sees RX(bias + Σ b_i w_i) since all RX
    # on one qubit commute and CRX acts as RX(w) iff control is 1 → P(1) = sin²(angle/2).
    w, b = (0.7, -0.4), 0.3
    for bits in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        circ = ir.Circuit(3)
        for q, bit in enumerate(bits):
            if bit:
                circ.append(ir.x(q))
        circ.extend(neurons.classic_neuron((0, 1), 2, w, b))
        angle = b + sum(bi * wi for bi, wi in zip(bits, w, strict=True))
        assert p_out1(circ, 2) == pytest.approx(np.sin(angle / 2) ** 2, abs=1e-12)


def test_prototype_neuron_zero_input_contributes() -> None:
    # The whole point of the prototype neuron [PTQNN §III.B]: a 0-valued input can still
    # activate the output (the classic neuron cannot: P(1)=sin²(bias/2) regardless of weights).
    circ = ir.Circuit(2)
    circ.extend(neurons.prototype_neuron((0,), 1, prototype=(np.pi,), theta=np.pi))
    assert p_out1(circ, 1) == pytest.approx(1.0, abs=1e-12)
    circ2 = ir.Circuit(2, [ir.x(0)])
    circ2.extend(neurons.prototype_neuron((0,), 1, prototype=(np.pi,), theta=np.pi))
    assert p_out1(circ2, 1) == pytest.approx(0.0, abs=1e-12)


def test_prototype_neuron_matches_pattern_01() -> None:
    # [PTQNN Fig 3 caption]: prototype |01⟩ maximal when q0=0, q1=1; reduced if one matches; zero
    # when the pattern is inverted. With φ=(π, 0) and θ=π/2 per input the activation is
    # P(1) = sin²(m·π/4) for m = number of matching inputs (RX(π/2) applied m times).
    proto, theta = (np.pi, 0.0), np.pi / 2
    expect = {(0, 1): 1.0, (0, 0): 0.5, (1, 1): 0.5, (1, 0): 0.0}
    for bits, e in expect.items():
        circ = ir.Circuit(3)
        for q, bit in enumerate(bits):
            if bit:
                circ.append(ir.x(q))
        circ.extend(neurons.prototype_neuron((0, 1), 2, proto, theta))
        assert p_out1(circ, 2) == pytest.approx(e, abs=1e-12)


def test_network_parameter_layout_and_counts() -> None:
    # Bit-count network [PTQNN Fig 5]: 5 inputs, 5 prototype hidden, 1 classic output → 11 qubits.
    net = neurons.Network(n_inputs=5, n_hidden=5, hidden="prototype")
    assert net.n_qubits == 11
    assert net.input_qubits == (0, 1, 2, 3, 4)
    assert net.hidden_qubits == (5, 6, 7, 8, 9)
    assert net.output_qubit == 10
    # hidden: 5 × (5 prototype angles + 1 θ) = 30; output classic: 5 weights + 1 bias = 6.
    assert net.n_params == 36
    classic = neurons.Network(n_inputs=5, n_hidden=5, hidden="classic")
    assert classic.n_params == 36  # "with the same amount of parameters" [PTQNN §III.C]


def test_network_circuit_structure() -> None:
    net = neurons.Network(n_inputs=2, n_hidden=2, hidden="prototype")
    params = np.arange(net.n_params, dtype=float) * 0.1
    circ = net.circuit(params)
    assert circ.n_qubits == 5
    # 2 hidden × 2 inputs CRX + 2 output CRX = 6 two-qubit gates
    assert circ.n_two_qubit_gates() == 6
    # prepend input preparation
    prep = [ir.h(0), ir.h(1)]
    circ2 = net.circuit(params, prep=prep)
    assert circ2.ops[:2] == prep or [o.name for o in circ2.ops[:2]] == ["h", "h"]
    with pytest.raises(ValueError, match="params"):
        net.circuit(params[:-1])


def test_network_classic_output_layer_reads_hidden() -> None:
    # Hidden neurons fully activated (θ=π, φ=π on |0⟩ inputs) and output CRX(π) from one hidden
    # qubit → P(out=1) = 1 for weight π on that synapse, 0 otherwise.
    net = neurons.Network(n_inputs=1, n_hidden=1, hidden="prototype")
    params = net.pack(hidden=[((np.pi,), np.pi)], output=((np.pi,), 0.0))
    assert p_out1(net.circuit(params), net.output_qubit) == pytest.approx(1.0, abs=1e-12)
    params = net.pack(hidden=[((np.pi,), np.pi)], output=((0.0,), 0.0))
    assert p_out1(net.circuit(params), net.output_qubit) == pytest.approx(0.0, abs=1e-12)
