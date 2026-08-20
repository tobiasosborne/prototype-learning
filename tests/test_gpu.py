"""GPU backends must agree with the CPU reference.  Skipped when CuPy/CUDA is unavailable."""

import numpy as np
import pytest

from ptqnn.backends import gpu, reference as ref
from ptqnn.backends.trajectories import TrajectoryBackend
from ptqnn.circuits import ir
from ptqnn.noise import channels as ch
from ptqnn.noise.model import NoiseModel
from ptqnn.tasks import bitcount

pytestmark = pytest.mark.skipif(not gpu.gpu_available(), reason="no usable CUDA device")


def _noisy() -> ir.Circuit:
    net = bitcount.network(3, 2)
    circ = net.circuit(
        net.random_params(np.random.default_rng(0)), prep=bitcount.superposition_prep(net)
    )
    nm = NoiseModel(
        one_qubit=ch.depolarizing(0.02, 1),
        two_qubit=ch.depolarizing(0.05, 2),
        t1_us=50.0,
        t2_us=70.0,
        duration_1q_ns=50.0,
        duration_2q_ns=400.0,
    )
    return nm.apply(circ)


def test_gpu_density_matrix_matches_cpu() -> None:
    circ = _noisy()
    np.testing.assert_allclose(
        gpu.GpuDensityMatrixBackend().probabilities(circ),
        ref.DensityMatrixBackend().probabilities(circ),
        atol=1e-12,
    )


def test_gpu_trajectories_match_density_matrix() -> None:
    circ = _noisy()
    exact = ref.DensityMatrixBackend().probabilities(circ)
    n = 20_000
    est = gpu.GpuTrajectoryBackend(n_trajectories=n, seed=1).probabilities(circ)
    sigma = np.sqrt(exact * (1 - exact) / n)
    assert np.all(np.abs(est - exact) < 4 * sigma + 1e-12)
    cpu = TrajectoryBackend(n_trajectories=200, seed=1).probabilities(circ)
    assert cpu.shape == est.shape


def test_gpu_unitary_and_initial_state() -> None:
    v = np.zeros(8, dtype=complex)
    v[5] = 1.0
    circ = ir.Circuit(3, [ir.crx(0, 1, 0.7)], initial_state=v)
    np.testing.assert_allclose(
        gpu.GpuDensityMatrixBackend().probabilities(circ),
        ref.StatevectorBackend().probabilities(circ),
        atol=1e-12,
    )
