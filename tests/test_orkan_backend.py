"""Contract tests for the orkan density-matrix backend against the numpy oracle.

``ptqnn.backends.reference.DensityMatrixBackend`` is the oracle (``CLAUDE.md``, "numpy reference
is the oracle; ``orkan``, ``trajectories``, ``cupy`` must agree with it").  Agreement is checked
on the *full* density matrix wherever the state is small enough, not only on the outcome
distribution — a backend that permutes qubits or drops coherences can still match a marginal
("test channels, not marginals").
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from ptqnn.backends import _orkan_ffi as ffi
from ptqnn.backends import reference
from ptqnn.backends.base import Backend
from ptqnn.backends.orkan import (
    OrkanDensityMatrixBackend,
    depolarizing_strength,
    unitary_mixture,
)
from ptqnn.circuits import ir
from ptqnn.noise import channels as ch
from ptqnn.noise.model import NoiseModel, NoisyBackend
from ptqnn.tasks import bitcount

pytestmark = [
    pytest.mark.orkan,
    pytest.mark.skipif(not ffi.orkan_available(), reason="liborkan.so not built/found"),
]

TOL = 1e-10


def haar_unitary(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(a)
    return np.asarray(q * (np.diag(r) / np.abs(np.diag(r))), dtype=np.complex128)


def random_circuit(n: int, seed: int, layers: int = 3, channels: bool = True) -> ir.Circuit:
    """rx/ry/rz/h/x/y/z + crx/cx + generic Haar unitaries + 1-qubit Kraus channels."""
    rng = np.random.default_rng(seed)
    circ = ir.Circuit(n)
    for layer in range(layers):
        for q in range(n):
            kind = int(rng.integers(0, 6))
            theta = float(rng.uniform(-np.pi, np.pi))
            circ.append(
                [ir.rx, ir.ry, ir.rz][kind](q, theta)
                if kind < 3
                else [ir.h, ir.x, ir.y][kind - 3](q)
            )
        # one generic (non-native) 1-qubit unitary per layer
        circ.append(ir.unitary([int(rng.integers(0, n))], haar_unitary(2, seed + layer), "u1"))
        for _ in range(n):
            c, t = (int(v) for v in rng.choice(n, size=2, replace=False))
            circ.append(ir.crx(c, t, float(rng.uniform(-np.pi, np.pi))))
        c, t = (int(v) for v in rng.choice(n, size=2, replace=False))
        circ.append(ir.cx(c, t))
        a, b = (int(v) for v in rng.choice(n, size=2, replace=False))
        circ.append(ir.unitary([a, b], haar_unitary(4, seed + 100 * layer), "u2"))
        if channels:
            circ.append(
                ir.Channel((int(rng.integers(0, n)),), ch.amplitude_damping(0.23), "amp_damp")
            )
            circ.append(
                ir.Channel((int(rng.integers(0, n)),), ch.depolarizing(0.17), "depolarizing")
            )
            circ.append(
                ir.Channel((int(rng.integers(0, n)),), ch.thermal_relaxation(60, 90, 0.4), "relax")
            )
    return circ


def random_state(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    return np.asarray(v / np.linalg.norm(v), dtype=np.complex128)


# ---------------------------------------------------------------------------
# Conventions
# ---------------------------------------------------------------------------


def test_satisfies_the_backend_protocol() -> None:
    be = OrkanDensityMatrixBackend()
    assert isinstance(be, Backend)
    assert be.name == "orkan-density-matrix"


def test_orkan_rx_convention_matches_ours() -> None:
    """orkan's native ``rx`` must be ``exp(-iθX/2)``, i.e. ``P(1) = sin²(θ/2)`` on |0⟩ ``[ir]``.

    The sign of θ is the whole risk: ``exp(+iθX/2)`` would still give the right probabilities
    from |0⟩ but the wrong state, so the check is on the density matrix.
    """
    be = OrkanDensityMatrixBackend()
    for theta in (0.7, -1.9, 2.5):
        circ = ir.Circuit(1, [ir.rx(0, theta)])
        got = be.density_matrix(circ)
        want = reference.run_density_matrix(circ)
        assert np.allclose(got, want, atol=TOL), theta
        assert abs(got[1, 1].real - np.sin(theta / 2) ** 2) < TOL


def test_native_and_generic_paths_agree() -> None:
    """Every gate routed to an orkan kernel must equal the same gate via ``two_from_mat``."""
    native = OrkanDensityMatrixBackend()
    generic = OrkanDensityMatrixBackend(native_gates=False)
    circ = random_circuit(4, seed=7)
    assert np.allclose(native.density_matrix(circ), generic.density_matrix(circ), atol=TOL)


# ---------------------------------------------------------------------------
# Agreement with the numpy oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [3, 4, 5])
def test_random_circuits_match_reference(n: int) -> None:
    be = OrkanDensityMatrixBackend()
    oracle = reference.DensityMatrixBackend()
    for seed in range(3):
        circ = random_circuit(n, seed=1000 * n + seed)
        assert np.allclose(be.probabilities(circ), oracle.probabilities(circ), atol=TOL)
        assert np.allclose(be.density_matrix(circ), reference.run_density_matrix(circ), atol=TOL)


@pytest.mark.parametrize("n", [3, 5])
def test_initial_state(n: int) -> None:
    be = OrkanDensityMatrixBackend()
    oracle = reference.DensityMatrixBackend()
    circ = random_circuit(n, seed=42)
    circ.initial_state = random_state(n, seed=n)
    assert np.allclose(be.probabilities(circ), oracle.probabilities(circ), atol=TOL)
    assert np.allclose(be.density_matrix(circ), reference.run_density_matrix(circ), atol=TOL)


def test_initial_state_alone_is_loaded_exactly() -> None:
    """|ψ⟩⟨ψ| written straight into the tiled buffer, with no ops at all."""
    be = OrkanDensityMatrixBackend()
    for n in (2, 5, 6):  # 6 > log2(TILE_DIM): several tiles, incl. off-diagonal ones
        psi = random_state(n, seed=n + 7)
        circ = ir.Circuit(n, initial_state=psi)
        assert np.allclose(be.density_matrix(circ), np.outer(psi, psi.conj()), atol=TOL)


# ---------------------------------------------------------------------------
# Two-qubit channels: mixture-of-unitaries fast path and the dense slow path
# ---------------------------------------------------------------------------


def two_qubit_depolarizing_circuit(n: int = 4) -> ir.Circuit:
    circ = random_circuit(n, seed=5, layers=2)
    circ.append(ir.Channel((1, 3), ch.depolarizing(0.21, 2), "depolarizing"))
    circ.append(ir.Channel((2, 0), ch.depolarizing(0.09, 2), "depolarizing"))
    circ.append(ir.crx(0, 2, 0.6))
    circ.append(ir.Channel((3, 1), ch.tensor(ch.pauli(0.1, 0.05, 0.2), ch.bit_flip(0.3)), "pauli"))
    return circ


@pytest.mark.parametrize("q", [0.0, 0.09, 0.5, 1.0])
@pytest.mark.parametrize("k", [1, 2])
def test_depolarizing_strength_recognises_depolarising(q: float, k: int) -> None:
    got = depolarizing_strength(ch.depolarizing(q, k))
    assert got is not None
    assert abs(got - q) < 1e-12


def test_depolarizing_strength_rejects_other_channels() -> None:
    """A biased Pauli channel and amplitude damping must not be mistaken for depolarising —
    the second has the same average fidelity as a depolarising channel of matched strength."""
    biased = ch.pauli(0.2, 0.05, 0.01)
    assert depolarizing_strength(biased) is None
    assert unitary_mixture(biased)  # …but it is still a mixture of unitaries
    damping = ch.amplitude_damping(0.3)
    assert depolarizing_strength(damping) is None
    assert unitary_mixture(damping) == []
    assert depolarizing_strength(ch.tensor(ch.depolarizing(0.1), ch.depolarizing(0.1))) is None


def test_two_qubit_mixture_of_unitaries_fast_path() -> None:
    be = OrkanDensityMatrixBackend()
    circ = two_qubit_depolarizing_circuit()
    assert np.allclose(be.density_matrix(circ), reference.run_density_matrix(circ), atol=TOL)
    assert be.stats["two_qubit_channels"] == 3
    # two depolarising channels take the twirl route, the biased Pauli one the mixture route
    assert be.stats["depolarizing_channels"] == 2
    assert be.stats["slow_path_channels"] == 0


def test_two_qubit_channel_slow_path_forced() -> None:
    be = OrkanDensityMatrixBackend(mixture_fast_path=False)
    circ = two_qubit_depolarizing_circuit()
    with pytest.warns(RuntimeWarning, match="slow path"):
        got = be.density_matrix(circ)
    assert np.allclose(got, reference.run_density_matrix(circ), atol=TOL)
    assert be.stats["slow_path_channels"] == 3


def test_two_qubit_non_unitary_channel_takes_slow_path() -> None:
    """Amplitude damping ⊗ depolarising is not a mixture of unitaries — dense round trip."""
    kraus = ch.tensor(ch.amplitude_damping(0.3), ch.depolarizing(0.12))
    circ = random_circuit(4, seed=11, layers=2)
    circ.append(ir.Channel((2, 0), kraus, "channel"))
    be = OrkanDensityMatrixBackend()
    with pytest.warns(RuntimeWarning, match="slow path"):
        got = be.density_matrix(circ)
    assert np.allclose(got, reference.run_density_matrix(circ), atol=TOL)
    assert be.stats["slow_path_channels"] == 1


# ---------------------------------------------------------------------------
# A whole noisy network
# ---------------------------------------------------------------------------


def noise_model() -> NoiseModel:
    return NoiseModel(
        one_qubit=ch.depolarizing(1e-3),
        two_qubit=ch.depolarizing(1.2e-2, 2),
        t1_us=90.0,
        t2_us=70.0,
        duration_1q_ns=35.0,
        duration_2q_ns=300.0,
        name="test",
    )


def test_noisy_bitcount_network_6_qubits() -> None:
    net = bitcount.network(3, 2)
    assert net.n_qubits == 6
    params = net.random_params(np.random.default_rng(3))
    circ = net.circuit(params, prep=bitcount.superposition_prep(net))
    nm = noise_model()
    got = NoisyBackend(OrkanDensityMatrixBackend(), nm).probabilities(circ)
    want = NoisyBackend(reference.DensityMatrixBackend(), nm).probabilities(circ)
    assert np.allclose(got, want, atol=1e-9)
    assert abs(got.sum() - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Lifetime
# ---------------------------------------------------------------------------


def test_unsupported_gate_raises_and_frees() -> None:
    circ = ir.Circuit(3, [ir.h(0)])
    circ.ops.append(ir.unitary([0, 1, 2], np.eye(8, dtype=np.complex128), "u3"))
    be = OrkanDensityMatrixBackend()
    for _ in range(5):
        with pytest.raises(NotImplementedError, match="3-qubit"):
            be.probabilities(circ)


def test_no_leak_over_many_calls() -> None:
    be = OrkanDensityMatrixBackend()
    circ = random_circuit(8, seed=2, layers=1)
    for _ in range(30):
        p = be.probabilities(circ)
    assert abs(p.sum() - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Timing (reported, never asserted)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_timing_11_qubits(capsys: pytest.CaptureFixture[str]) -> None:
    """Reported, never asserted: seconds per ``probabilities`` call on the 11-qubit noisy
    bit-count network ``[PTQNN Fig 5]``.

    Marked ``slow`` because of the *numpy* leg: one density-matrix pass over ~300 ops at
    ``n = 11`` moves a few hundred GB through ``np.tensordot`` transposes and takes ~15 min,
    against seconds for orkan.  That ratio is the whole point of this backend.
    """
    net = bitcount.network(5, 5)
    assert net.n_qubits == 11
    params = net.random_params(np.random.default_rng(0))
    circ = net.circuit(params, prep=bitcount.superposition_prep(net))
    nm = noise_model()
    noisy = nm.apply(circ)
    n_1q = sum(1 for op in noisy.ops if len(op.qubits) == 1)
    n_2q = len(noisy.ops) - n_1q

    orkan_be = OrkanDensityMatrixBackend()
    t0 = time.perf_counter()
    p_orkan = orkan_be.probabilities(noisy)
    t_orkan = time.perf_counter() - t0

    t0 = time.perf_counter()
    p_numpy = reference.DensityMatrixBackend().probabilities(noisy)
    t_numpy = time.perf_counter() - t0

    with capsys.disabled():
        print(
            f"\n[11q noisy bitcount(5,5): {len(noisy.ops)} ops = {n_1q} 1q + {n_2q} 2q, "
            f"{orkan_be.stats['two_qubit_channels']} 2q channels "
            f"({orkan_be.stats['slow_path_channels']} slow path)]\n"
            f"  orkan MIXED_TILED : {t_orkan:8.3f} s per probabilities()\n"
            f"  numpy density mat : {t_numpy:8.3f} s per probabilities()\n"
            f"  speed-up          : {t_numpy / t_orkan:8.1f}x\n"
            f"  max |Δp|          : {np.abs(p_orkan - p_numpy).max():.2e}"
        )
