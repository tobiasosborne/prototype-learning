"""``ptqnn.backends.batched`` — one batched statevector kernel for the three task fast paths
(review 002 finding 7).  Pinned to the numpy reference on full distributions, for every gate
shape the IR can build (1q, controlled-1q, generic 2q, SWAP), not only rx/crx."""

from __future__ import annotations

import numpy as np
import pytest

from ptqnn.backends import batched
from ptqnn.backends import reference as ref
from ptqnn.circuits import ir
from ptqnn.circuits.neurons import Network
from ptqnn.hardware import transpile as tp
from ptqnn.noise import channels as ch


def _random_circuit(n: int, rng: np.random.Generator, n_ops: int = 30) -> ir.Circuit:
    c = ir.Circuit(n)
    for _ in range(n_ops):
        kind = rng.integers(5)
        q0, q1 = rng.choice(n, size=2, replace=False)
        if kind == 0:
            c.append(ir.rx(int(q0), float(rng.uniform(-np.pi, np.pi))))
        elif kind == 1:
            c.append(ir.h(int(q0)))
        elif kind == 2:
            c.append(ir.crx(int(q0), int(q1), float(rng.uniform(-np.pi, np.pi))))
        elif kind == 3:
            c.append(tp.swap(int(q0), int(q1)))
        else:
            # generic 2q unitary: exp(-i θ ZZ/2) composed with a random 1q pair, via QR
            a = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
            u, _ = np.linalg.qr(a)
            c.append(ir.Gate((int(q0), int(q1)), u, "u4"))
    return c


@pytest.mark.parametrize("n", [3, 5])
def test_batch_agrees_with_reference_on_random_states(n: int) -> None:
    rng = np.random.default_rng(n)
    circ = _random_circuit(n, rng)
    states = rng.normal(size=(7, 2**n)) + 1j * rng.normal(size=(7, 2**n))
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    out = batched.run_batch(circ, states)
    for b in range(7):
        want = ref.run_statevector(circ, states[b])
        assert np.allclose(out[b], want, atol=1e-12)


def test_backend_protocol_probabilities_match_reference() -> None:
    rng = np.random.default_rng(1)
    net = Network(n_inputs=3, n_hidden=2, hidden="prototype")
    circ = net.circuit(net.random_params(rng), [ir.h(q) for q in net.input_qubits])
    be = batched.BatchedStatevectorBackend()
    assert be.name
    assert np.allclose(be.probabilities(circ), ref.StatevectorBackend().probabilities(circ))


def test_basis_columns_are_the_evolved_input_basis() -> None:
    rng = np.random.default_rng(2)
    net = Network(n_inputs=2, n_hidden=3, hidden="prototype")
    tail = net.circuit(net.random_params(rng))
    cols = batched.basis_columns(tail, net.input_qubits)
    n = net.n_qubits
    assert cols.shape == (2**n, 4)
    for b in range(4):
        psi0 = np.zeros(2**n, dtype=np.complex128)
        idx = sum(((b >> (1 - q)) & 1) << (n - 1 - q) for q in net.input_qubits)
        psi0[idx] = 1.0
        assert np.allclose(cols[:, b], ref.run_statevector(tail, psi0), atol=1e-12)


def test_channels_are_refused_loudly() -> None:
    circ = ir.Circuit(2, ops=[ir.h(0), ir.Channel((0,), ch.depolarizing(0.1, 1), "dep")])
    with pytest.raises(TypeError, match="unitary-only"):
        batched.run_batch(circ, np.eye(4, dtype=np.complex128)[:1])


def test_shape_errors_are_loud() -> None:
    circ = ir.Circuit(2, ops=[ir.h(0)])
    with pytest.raises(ValueError):
        batched.run_batch(circ, np.ones((1, 3), dtype=np.complex128))
