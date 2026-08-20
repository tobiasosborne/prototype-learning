"""Red-phase tests for the Monte-Carlo Kraus-unravelling backend (``backends.trajectories``)."""

import numpy as np
import pytest

from ptqnn.backends import reference as ref
from ptqnn.backends.trajectories import TrajectoryBackend
from ptqnn.circuits import ir
from ptqnn.noise import channels as ch
from ptqnn.noise.model import NoiseModel, NoisyBackend


def noisy_circuit() -> ir.Circuit:
    circ = ir.Circuit(3, [ir.h(0), ir.crx(0, 1, 1.1), ir.rx(2, 0.7), ir.crx(2, 1, 2.0)])
    nm = NoiseModel(
        one_qubit=ch.depolarizing(0.1, 1),
        two_qubit=ch.depolarizing(0.2, 2),
        t1_us=10.0,
        t2_us=15.0,
        duration_1q_ns=500.0,
        duration_2q_ns=2000.0,
    )
    return nm.apply(circ)


def test_unitary_circuit_is_exact_with_one_trajectory() -> None:
    circ = ir.Circuit(3, [ir.h(0), ir.crx(0, 2, 0.9), ir.rx(1, 0.4)])
    be = TrajectoryBackend(n_trajectories=1, seed=0)
    np.testing.assert_allclose(
        be.probabilities(circ), ref.StatevectorBackend().probabilities(circ), atol=1e-13
    )
    assert be.name.startswith("trajectories")


def test_matches_density_matrix_within_statistical_error() -> None:
    circ = noisy_circuit()
    exact = ref.DensityMatrixBackend().probabilities(circ)
    n = 20_000
    est = TrajectoryBackend(n_trajectories=n, seed=1, batch_size=4096).probabilities(circ)
    sigma = np.sqrt(exact * (1 - exact) / n)
    assert np.all(np.abs(est - exact) < 4 * sigma + 1e-12)
    assert est.sum() == pytest.approx(1.0)


def test_seeded_and_batch_invariant() -> None:
    circ = noisy_circuit()
    a = TrajectoryBackend(n_trajectories=500, seed=3, batch_size=100).probabilities(circ)
    b = TrajectoryBackend(n_trajectories=500, seed=3, batch_size=100).probabilities(circ)
    assert np.array_equal(a, b)


def test_initial_state_and_noisy_backend_wrapper() -> None:
    v = np.zeros(8, dtype=complex)
    v[0b101] = 1.0
    circ = ir.Circuit(3, [ir.x(1)], initial_state=v)
    p = TrajectoryBackend(n_trajectories=3, seed=0).probabilities(circ)
    assert p[0b111] == pytest.approx(1.0)
    nm = NoiseModel(readout=ch.ReadoutError(0.1, 0.0))
    p2 = NoisyBackend(TrajectoryBackend(n_trajectories=3, seed=0), nm).probabilities(circ)
    assert p2[0b111] == pytest.approx(1.0)  # readout p01 only flips zeros → none here
    circ0 = ir.Circuit(1, [])
    np.testing.assert_allclose(
        NoisyBackend(TrajectoryBackend(n_trajectories=3, seed=0), nm).probabilities(circ0),
        [0.9, 0.1],
    )


def test_sampling_uses_kraus_branch_probabilities() -> None:
    # Amplitude damping with γ=1 on |1⟩ must deterministically give |0⟩ (single branch has p=1).
    circ = ir.Circuit(1, [ir.x(0), ir.Channel((0,), ch.amplitude_damping(1.0), "ad")])
    p = TrajectoryBackend(n_trajectories=50, seed=0).probabilities(circ)
    np.testing.assert_allclose(p, [1.0, 0.0], atol=1e-13)
