"""Red-phase tests for the numpy reference backend (``ptqnn.backends.reference``).

All expectations are built with explicit ``np.kron`` products (msb-first ordering) inside this
file, independently of the backend's tensor-reshape machinery.
"""

from functools import reduce

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ptqnn.backends import reference as ref
from ptqnn.circuits import ir

I2 = np.eye(2, dtype=complex)
P0 = np.diag([1.0, 0.0]).astype(complex)
P1 = np.diag([0.0, 1.0]).astype(complex)


def kron_all(ms: list[np.ndarray]) -> np.ndarray:
    return reduce(np.kron, ms)


def one_site(n: int, q: int, m: np.ndarray) -> np.ndarray:
    return kron_all([m if i == q else I2 for i in range(n)])


def crx_full(n: int, c: int, t: int, th: float) -> np.ndarray:
    rxm = ir.rx(0, th).matrix
    return one_site(n, c, P0) + kron_all(
        [P1 if i == c else (rxm if i == t else I2) for i in range(n)]
    )


def test_zero_state_shapes() -> None:
    psi = ref.zero_state(3)
    assert psi.shape == (8,)
    assert psi[0] == 1
    rho = ref.zero_density_matrix(3)
    assert rho.shape == (8, 8)
    assert rho[0, 0] == 1 and np.trace(rho) == 1


def test_single_qubit_gate_statevector_msb_first() -> None:
    # X on qubit 0 of |000⟩ must give |100⟩ = index 4 (qubit 0 is the MSB).
    c = ir.Circuit(3, [ir.x(0)])
    psi = ref.run_statevector(c)
    assert psi[4] == pytest.approx(1.0)


def test_crx_statevector_against_kron() -> None:
    n, th = 4, 0.83
    psi0 = np.zeros(2**n, dtype=complex)
    psi0[0b1010] = 1.0  # qubit 0 = 1, qubit 2 = 1
    for c, t in [(0, 3), (3, 0), (2, 1), (1, 2)]:
        circ = ir.Circuit(n, [ir.crx(c, t, th)])
        got = ref.run_statevector(circ, psi0)
        np.testing.assert_allclose(got, crx_full(n, c, t, th) @ psi0, atol=1e-14)


def test_density_matrix_matches_statevector_for_unitary_circuit() -> None:
    n = 4
    rng = np.random.default_rng(0)
    circ = ir.Circuit(n)
    for q in range(n):
        circ.append(ir.h(q))
    for _ in range(6):
        c, t = rng.choice(n, 2, replace=False)
        circ.append(ir.crx(int(c), int(t), float(rng.uniform(-np.pi, np.pi))))
        circ.append(ir.rx(int(t), float(rng.uniform(-np.pi, np.pi))))
    psi = ref.run_statevector(circ)
    rho = ref.run_density_matrix(circ)
    np.testing.assert_allclose(rho, np.outer(psi, psi.conj()), atol=1e-13)
    np.testing.assert_allclose(ref.probabilities(psi), ref.probabilities(rho), atol=1e-13)


def test_kraus_channel_matches_explicit_sum() -> None:
    n, gamma = 3, 0.3
    k0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
    k1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
    circ = ir.Circuit(n, [ir.h(0), ir.h(1), ir.h(2), ir.Channel((1,), [k0, k1], "ad")])
    rho = ref.run_density_matrix(circ)
    plus = np.full(8, 1 / np.sqrt(8), dtype=complex)
    rho_in = np.outer(plus, plus.conj())
    expected = sum(one_site(n, 1, k) @ rho_in @ one_site(n, 1, k).conj().T for k in (k0, k1))
    np.testing.assert_allclose(rho, expected, atol=1e-14)
    assert np.trace(rho) == pytest.approx(1.0)


def test_two_qubit_kraus_channel_on_reversed_qubits() -> None:
    # A 2-qubit depolarising-like channel with nontrivial ordering: K = CRX(θ) applied on (2, 0).
    n, th = 3, 0.5
    u = ir.crx(0, 1, th).matrix  # generic 4x4 in (control, target) order
    ch = ir.Channel((2, 0), [u * np.sqrt(0.5), np.eye(4) * np.sqrt(0.5)], "mix")
    rng = np.random.default_rng(1)
    v = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    v /= np.linalg.norm(v)
    rho_in = np.outer(v, v.conj())
    got = ref.apply_channel(rho_in, ch, n)
    full_u = crx_full(n, 2, 0, th)
    expected = 0.5 * full_u @ rho_in @ full_u.conj().T + 0.5 * rho_in
    np.testing.assert_allclose(got, expected, atol=1e-13)


def test_marginal_probabilities_and_joint() -> None:
    # |+⟩|0⟩|1⟩: P(q0=1)=1/2, P(q2=1)=1, joint over (0,2) = {01: 1/2, 11: 1/2}.
    circ = ir.Circuit(3, [ir.h(0), ir.x(2)])
    p = ref.probabilities(ref.run_statevector(circ))
    np.testing.assert_allclose(ref.marginal(p, [0]), [0.5, 0.5])
    np.testing.assert_allclose(ref.marginal(p, [2]), [0.0, 1.0])
    np.testing.assert_allclose(ref.marginal(p, [0, 2]), [0.0, 0.5, 0.0, 0.5])


def test_sampling_is_seeded_and_unbiased() -> None:
    p = np.array([0.1, 0.2, 0.3, 0.4])
    counts = ref.sample_counts(p, shots=20_000, rng=np.random.default_rng(7))
    assert counts.sum() == 20_000
    np.testing.assert_allclose(counts / 20_000, p, atol=0.02)
    again = ref.sample_counts(p, shots=20_000, rng=np.random.default_rng(7))
    assert np.array_equal(counts, again)


@settings(max_examples=25, deadline=None)
@given(st.integers(min_value=1, max_value=4), st.integers(min_value=0, max_value=2**31 - 1))
def test_random_circuit_trace_and_hermiticity(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    circ = ir.Circuit(n)
    for _ in range(8):
        q = int(rng.integers(n))
        circ.append(ir.rx(q, float(rng.uniform(-np.pi, np.pi))))
        if n > 1 and rng.random() < 0.5:
            c, t = rng.choice(n, 2, replace=False)
            circ.append(ir.crx(int(c), int(t), float(rng.uniform(-np.pi, np.pi))))
        if rng.random() < 0.3:
            g = float(rng.uniform(0, 1))
            circ.append(
                ir.Channel(
                    (q,),
                    [np.diag([1, np.sqrt(1 - g)]), np.array([[0, np.sqrt(g)], [0, 0]])],
                    "ad",
                )
            )
    rho = ref.run_density_matrix(circ)
    assert np.trace(rho).real == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(rho, rho.conj().T, atol=1e-12)
    assert np.linalg.eigvalsh(rho).min() > -1e-10
