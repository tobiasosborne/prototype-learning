"""Red-phase tests for the CP-map library (``ptqnn.noise.channels``).

Every channel is checked at the Choi level (CP + TP), not just on one input state.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ptqnn.noise import channels as ch

I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0, -1.0]).astype(complex)


def assert_cptp(ks: list[np.ndarray]) -> None:
    d = ks[0].shape[0]
    np.testing.assert_allclose(sum(k.conj().T @ k for k in ks), np.eye(d), atol=1e-12)
    c = ch.choi(ks)
    np.testing.assert_allclose(c, c.conj().T, atol=1e-12)
    assert np.linalg.eigvalsh(c).min() > -1e-12


def test_choi_of_identity_is_maximally_entangled_projector() -> None:
    c = ch.choi([np.eye(2, dtype=complex)])
    bell = np.array([1, 0, 0, 1], dtype=complex)
    np.testing.assert_allclose(c, np.outer(bell, bell.conj()), atol=1e-14)
    assert np.trace(c).real == pytest.approx(2.0)  # Tr Choi = d for TP maps


def test_depolarizing_kraus_and_fidelity() -> None:
    # E(ρ) = (1−p)ρ + p I/d. For a qubit: F_avg = 1 − p/2; for two qubits: 1 − 3p/4.
    for n, p in [(1, 0.1), (2, 0.05)]:
        ks = ch.depolarizing(p, n_qubits=n)
        assert_cptp(ks)
        rho = np.zeros((2**n, 2**n), dtype=complex)
        rho[0, 0] = 1
        out = ch.apply(ks, rho)
        np.testing.assert_allclose(out, (1 - p) * rho + p * np.eye(2**n) / 2**n, atol=1e-12)
        d = 2**n
        assert ch.average_gate_fidelity(ks) == pytest.approx(1 - p * (d - 1) / d)
    # inverse: from average fidelity back to p
    assert ch.depolarizing_probability(0.999, 1) == pytest.approx(0.002)
    assert ch.depolarizing_probability(0.99, 2) == pytest.approx(0.04 / 3)
    assert ch.average_gate_fidelity(ch.depolarizing_from_fidelity(0.99, 2)) == pytest.approx(0.99)


def test_amplitude_and_phase_damping() -> None:
    g = 0.3
    ks = ch.amplitude_damping(g)
    assert_cptp(ks)
    rho1 = np.diag([0.0, 1.0]).astype(complex)
    np.testing.assert_allclose(ch.apply(ks, rho1), np.diag([g, 1 - g]), atol=1e-12)
    lam = 0.4
    kp = ch.phase_damping(lam)
    assert_cptp(kp)
    plus = np.full((2, 2), 0.5, dtype=complex)
    out = ch.apply(kp, plus)
    assert out[0, 1] == pytest.approx(0.5 * np.sqrt(1 - lam))
    np.testing.assert_allclose(np.diag(out), [0.5, 0.5])


def test_thermal_relaxation_matches_t1_t2_decay() -> None:
    # After time t: populations relax with exp(−t/T1), coherences with exp(−t/T2) (T2 ≤ 2 T1).
    t1, t2, t = 100.0, 80.0, 25.0
    ks = ch.thermal_relaxation(t1_us=t1, t2_us=t2, duration_us=t)
    assert_cptp(ks)
    rho = np.array([[0.3, 0.4], [0.4, 0.7]], dtype=complex)
    out = ch.apply(ks, rho)
    assert out[1, 1].real == pytest.approx(0.7 * np.exp(-t / t1))
    assert abs(out[0, 1]) == pytest.approx(0.4 * np.exp(-t / t2))
    with pytest.raises(ValueError, match="T2"):
        ch.thermal_relaxation(t1_us=10, t2_us=25, duration_us=1)
    # excited-state population at equilibrium
    ks_hot = ch.thermal_relaxation(t1_us=t1, t2_us=t2, duration_us=1e6, excited_population=0.1)
    out = ch.apply(ks_hot, np.diag([1.0, 0.0]).astype(complex))
    assert out[1, 1].real == pytest.approx(0.1, abs=1e-6)


def test_pauli_channel_and_flips() -> None:
    ks = ch.pauli(px=0.1, py=0.2, pz=0.3)
    assert_cptp(ks)
    rho = np.array([[0.6, 0.2], [0.2, 0.4]], dtype=complex)
    expected = 0.4 * rho + 0.1 * X @ rho @ X + 0.2 * Y @ rho @ Y + 0.3 * Z @ rho @ Z
    np.testing.assert_allclose(ch.apply(ks, rho), expected, atol=1e-12)
    assert len(ch.bit_flip(0.05)) == 2
    assert len(ch.phase_flip(0.05)) == 2
    with pytest.raises(ValueError):
        ch.pauli(0.5, 0.5, 0.5)


def test_coherent_overrotation_is_unitary_single_kraus() -> None:
    ks = ch.coherent_overrotation("x", epsilon=0.05)
    assert len(ks) == 1
    np.testing.assert_allclose(ks[0].conj().T @ ks[0], I2, atol=1e-14)
    # average fidelity of a small over-rotation ≈ 1 − ε²/6 for a qubit
    assert ch.average_gate_fidelity(ks) == pytest.approx(1 - 0.05**2 / 6, abs=1e-5)


def test_compose_and_tensor() -> None:
    a = ch.amplitude_damping(0.2)
    b = ch.phase_damping(0.3)
    comp = ch.compose(b, a)  # apply a first, then b
    assert_cptp(comp)
    rho = np.array([[0.3, 0.4], [0.4, 0.7]], dtype=complex)
    np.testing.assert_allclose(ch.apply(comp, rho), ch.apply(b, ch.apply(a, rho)), atol=1e-12)
    t = ch.tensor(a, b)
    assert t[0].shape == (4, 4)
    assert_cptp(t)
    rho2 = np.kron(rho, np.diag([0.5, 0.5]))
    np.testing.assert_allclose(
        ch.apply(t, rho2), np.kron(ch.apply(a, rho), ch.apply(b, np.diag([0.5, 0.5]))), atol=1e-12
    )


def test_choi_to_kraus_round_trip_and_minimal_rank() -> None:
    ks = ch.depolarizing(0.2, 1)  # 4 Kraus ops
    ks2 = ch.choi_to_kraus(ch.choi(ks))
    assert len(ks2) == 4
    rho = np.array([[0.3, 0.4j], [-0.4j, 0.7]], dtype=complex)
    np.testing.assert_allclose(ch.apply(ks2, rho), ch.apply(ks, rho), atol=1e-12)
    ks3 = ch.choi_to_kraus(
        ch.choi(ch.compose(ch.amplitude_damping(0.1), ch.amplitude_damping(0.2)))
    )
    assert len(ks3) == 2  # composing two rank-2 damping channels is again rank 2 (same fixed point)


def test_readout_confusion_matrix() -> None:
    # p01 = P(read 1 | true 0), p10 = P(read 0 | true 1). Applied to the diagonal (probabilities).
    r = ch.ReadoutError(p01=0.02, p10=0.05)
    np.testing.assert_allclose(r.confusion, [[0.98, 0.05], [0.02, 0.95]])
    p = np.array([1.0, 0.0])
    np.testing.assert_allclose(r.apply(p, n_qubits=1), [0.98, 0.02])
    # two qubits, independent readout: P(read 11 | true 00) = p01² etc.
    p2 = np.array([1.0, 0, 0, 0])
    out = r.apply(p2, n_qubits=2)
    assert out[3] == pytest.approx(0.02**2)
    assert out.sum() == pytest.approx(1.0)
    # per-qubit list
    r2 = ch.ReadoutError.per_qubit([(0.02, 0.05), (0.0, 0.0)])
    out = r2.apply(p2, n_qubits=2)
    np.testing.assert_allclose(out, [0.98, 0.0, 0.02, 0.0])


@settings(max_examples=30, deadline=None)
@given(
    st.floats(0, 1),
    st.floats(0, 1),
    st.floats(1, 500),
    st.floats(0.01, 1.0),
    st.floats(0, 50),
)
def test_all_channels_are_cptp(p: float, g: float, t1: float, ratio: float, dt: float) -> None:
    assert_cptp(ch.depolarizing(p, 1))
    assert_cptp(ch.amplitude_damping(g))
    assert_cptp(ch.phase_damping(g))
    assert_cptp(ch.thermal_relaxation(t1_us=t1, t2_us=ratio * 2 * t1, duration_us=dt))
