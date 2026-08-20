"""Red-phase tests for losses, the KW–SA optimiser and the backend protocol."""

import numpy as np
import pytest

from ptqnn.backends import reference as ref
from ptqnn.backends.base import Backend
from ptqnn.circuits import ir
from ptqnn.train import kwsa, losses


def test_kl_divergence_definition_and_edge_cases() -> None:
    # D_KL(P‖Q) = Σ p log(p/q) [PTQNN Eq 4]; zero iff equal; p=0 terms contribute 0.
    p = np.array([0.5, 0.5, 0.0])
    q = np.array([0.25, 0.5, 0.25])
    assert losses.kl_divergence(p, p) == pytest.approx(0.0)
    assert losses.kl_divergence(p, q) == pytest.approx(0.5 * np.log(2))
    assert losses.kl_divergence(p, q) >= 0
    # q=0 where p>0 → large finite value (eps-regularised), not inf/nan
    big = losses.kl_divergence(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert np.isfinite(big) and big > 10


def test_cosine_dissimilarity() -> None:
    # D_cos(P,Q) = 1 − P·Q for unit vectors [PTQNN Eq 5]
    p = np.array([1.0, 0.0])
    assert losses.cosine_dissimilarity(p, p) == pytest.approx(0.0)
    assert losses.cosine_dissimilarity(p, np.array([0.0, 1.0])) == pytest.approx(1.0)
    assert losses.cosine_dissimilarity(np.array([2.0, 0.0]), p) == pytest.approx(0.0)  # normalised


def test_kwsa_minimises_quadratic_deterministically() -> None:
    target = np.array([0.3, -1.2, 2.0])

    def f(x: np.ndarray) -> float:
        return float(np.sum((x - target) ** 2))

    cfg = kwsa.KWSAConfig(iterations=200, a0=0.2, c0=0.1, t0=0.0, seed=1, mode="full")
    res = kwsa.minimise(f, np.zeros(3), cfg)
    assert res.loss < 1e-3
    np.testing.assert_allclose(res.params, target, atol=0.05)
    assert len(res.history) == 201  # initial + one per iteration
    assert res.history[0] == pytest.approx(f(np.zeros(3)))
    again = kwsa.minimise(f, np.zeros(3), cfg)
    assert np.array_equal(res.params, again.params)


def test_kwsa_accepts_uphill_moves_only_when_hot() -> None:
    rng = np.random.default_rng(0)
    # Metropolis rule [PTQNN Eq 3]: P = exp(−Δe/(kT)); Δe<0 always accepted.
    assert kwsa.accept(delta=-0.1, temperature=0.0, k=1.0, rng=rng)
    assert kwsa.accept(delta=-0.1, temperature=1.0, k=1.0, rng=rng)
    assert not kwsa.accept(delta=0.1, temperature=0.0, k=1.0, rng=rng)
    hits = sum(kwsa.accept(0.1, temperature=1.0, k=1.0, rng=rng) for _ in range(4000))
    assert hits / 4000 == pytest.approx(np.exp(-0.1), abs=0.03)


def test_kwsa_escapes_local_minimum_with_annealing() -> None:
    # Double well with a shallow local min at x=−1 (f=0.5) and global at x=+1.2 (f=0).
    def f(x: np.ndarray) -> float:
        t = float(x[0])
        return min((t + 1) ** 2 + 0.5, (t - 1.2) ** 2)

    cold = kwsa.minimise(
        f, np.array([-1.0]), kwsa.KWSAConfig(iterations=300, t0=0.0, seed=3, mode="full")
    )
    hot = kwsa.minimise(
        f,
        np.array([-1.0]),
        kwsa.KWSAConfig(iterations=400, t0=1.0, cooling=0.99, jitter=1.0, seed=3, mode="full"),
    )
    assert cold.loss == pytest.approx(0.5, abs=0.05)
    assert hot.loss < 0.05


def test_kwsa_coordinate_mode_minimises_quadratic() -> None:
    target = np.array([0.3, -1.2, 2.0])

    def f(x: np.ndarray) -> float:
        return float(np.sum((x - target) ** 2))

    res = kwsa.minimise(f, np.zeros(3), kwsa.KWSAConfig(iterations=30, a0=0.5, c0=0.1, seed=1))
    assert res.loss < 1e-4
    assert res.evaluations == 1 + 30 * 3 * 3


def test_backends_expose_probabilities_protocol() -> None:
    circ = ir.Circuit(2, [ir.h(0), ir.crx(0, 1, np.pi)])
    sv: Backend = ref.StatevectorBackend()
    dm: Backend = ref.DensityMatrixBackend()
    np.testing.assert_allclose(sv.probabilities(circ), [0.5, 0, 0, 0.5], atol=1e-14)
    np.testing.assert_allclose(dm.probabilities(circ), [0.5, 0, 0, 0.5], atol=1e-14)
    assert sv.name == "numpy-statevector"
