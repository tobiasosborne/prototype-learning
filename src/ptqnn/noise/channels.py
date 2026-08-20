"""Completely positive maps as Kraus sets, with Choi-level algebra and the standard noise zoo.

All functions work on plain lists of ``2^k × 2^k`` complex arrays (Kraus operators ``K_i`` with
``Σ K_i† K_i = I``) so they compose freely and can be wrapped into
:class:`ptqnn.circuits.ir.Channel` once a qubit assignment is known.  Conventions:

* Choi matrix ``C = Σ_ij |i⟩⟨j| ⊗ E(|i⟩⟨j|)`` (unnormalised, ``Tr C = d`` for TP maps).  CP ⇔
  ``C ≥ 0``; TP ⇔ ``Tr_out C = I``.  Channels are compared through ``C``, never through a single
  output state (the marginal-test lesson from ``../Sturm.jl``).
* Entanglement (process) fidelity ``F_e = Σ_i |Tr K_i|² / d²`` and average gate fidelity
  ``F_avg = (d F_e + 1)/(d + 1)`` [Nielsen, Phys. Lett. A 303, 249 (2002), Eq. 3 — standard
  relation, no local PDF; marked as such].
* Depolarising ``E(ρ) = (1−p) ρ + p I/d`` has ``F_avg = 1 − p (d−1)/d``, hence vendor-quoted
  average error ``ε = 1 − F_avg`` maps to ``p = ε d/(d−1)`` (``2ε`` for one qubit, ``4ε/3`` for
  two).  Qiskit-Aer's ``depolarizing_error`` uses the same ``p``.
* Thermal relaxation for duration ``t`` with ``T1``, ``T2`` (``T2 ≤ 2 T1`` for complete
  positivity): populations relax as ``exp(−t/T1)`` towards the equilibrium excited population
  ``n_eq``; coherences decay as ``exp(−t/T2)``.  For ``T2 > T1`` the map is not a Pauli channel,
  so the Kraus set is obtained from the Choi matrix by eigendecomposition (``choi_to_kraus``).

The Kraus forms of amplitude/phase damping follow Nielsen & Chuang §8.3.5–8.3.6 (standard;
quoted in ``[SNN §IV.A]`` as "amplitude damping ... decay rate of a single qubit from the excited
state |1⟩ to the ground state |0⟩").
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Complex = npt.NDArray[np.complex128]
Real = npt.NDArray[np.float64]
Kraus = list[Complex]

_ATOL = 1e-12

_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
PAULIS = {"i": _I, "x": _X, "y": _Y, "z": _Z}


def _c(m: npt.ArrayLike) -> Complex:
    return np.asarray(m, dtype=np.complex128)


# --- algebra -------------------------------------------------------------------------------------


def apply(ks: Sequence[Complex], rho: Complex) -> Complex:
    """``Σ K ρ K†`` on a full density matrix of the channel's own dimension."""
    return sum((k @ rho @ k.conj().T for k in ks), np.zeros_like(_c(rho)))


def is_trace_preserving(ks: Sequence[Complex], atol: float = 1e-10) -> bool:
    d = ks[0].shape[0]
    return bool(np.allclose(sum(k.conj().T @ k for k in ks), np.eye(d), atol=atol))


def choi(ks: Sequence[Complex]) -> Complex:
    """``C = Σ_ij |i⟩⟨j| ⊗ E(|i⟩⟨j|)``, i.e. ``C = Σ_k vec(K_k) vec(K_k)†`` with row-major
    vectorisation ``vec(K)[i·d + o] = K[o, i]``."""
    d = ks[0].shape[0]
    c = np.zeros((d * d, d * d), dtype=np.complex128)
    for k in ks:
        v = k.T.reshape(-1)  # v[i*d + o] = K[o, i]
        c += np.outer(v, v.conj())
    return c


def choi_to_kraus(c: Complex, tol: float = 1e-12) -> Kraus:
    """Minimal Kraus set from a Choi matrix by eigendecomposition (rank = #Kraus ops)."""
    d = round(np.sqrt(c.shape[0]))
    w, v = np.linalg.eigh(c)
    if w.min() < -1e-9:
        raise ValueError(f"Choi matrix is not positive (min eigenvalue {w.min():.3e}): not CP")
    ks: Kraus = []
    for lam, vec in zip(w, v.T, strict=True):
        if lam > tol:
            ks.append(np.sqrt(lam) * vec.reshape(d, d).T)
    return ks


def compose(second: Sequence[Complex], first: Sequence[Complex]) -> Kraus:
    """Kraus set of ``second ∘ first`` (apply ``first``, then ``second``)."""
    return [b @ a for b in second for a in first]


def tensor(a: Sequence[Complex], b: Sequence[Complex]) -> Kraus:
    """``A ⊗ B`` with ``a`` on the more significant factor."""
    return [_c(np.kron(ka, kb)) for ka in a for kb in b]


def entanglement_fidelity(ks: Sequence[Complex]) -> float:
    d = ks[0].shape[0]
    return float(sum(abs(np.trace(k)) ** 2 for k in ks) / d**2)


def average_gate_fidelity(ks: Sequence[Complex]) -> float:
    d = ks[0].shape[0]
    return float((d * entanglement_fidelity(ks) + 1) / (d + 1))


# --- the zoo -------------------------------------------------------------------------------------


def depolarizing(p: float, n_qubits: int = 1) -> Kraus:
    """``E(ρ) = (1−p) ρ + p I/d`` as ``√(1−p+p/d²) I`` plus ``√(p/d²) P`` for the non-identity
    Paulis ``P`` (``d = 2^n``)."""
    if not 0 <= p <= 1:
        raise ValueError("p must lie in [0, 1]")
    d = 2**n_qubits
    paulis_1q = [_I, _X, _Y, _Z]
    strings: list[Complex] = [np.array([[1.0]], dtype=np.complex128)]
    for _ in range(n_qubits):
        strings = [_c(np.kron(s, q)) for s in strings for q in paulis_1q]
    ks = [np.sqrt(1 - p + p / d**2) * strings[0]]
    ks += [np.sqrt(p / d**2) * s for s in strings[1:]]
    return ks


def depolarizing_probability(f_avg: float, n_qubits: int) -> float:
    """``p = (1 − F_avg) d/(d−1)`` — the depolarising strength with average gate fidelity ``F``."""
    d = 2**n_qubits
    return float((1 - f_avg) * d / (d - 1))


def depolarizing_from_fidelity(f_avg: float, n_qubits: int) -> Kraus:
    """Depolarising channel with the given average gate fidelity."""
    return depolarizing(depolarizing_probability(f_avg, n_qubits), n_qubits)


def amplitude_damping(gamma: float) -> Kraus:
    """``K0 = diag(1, √(1−γ))``, ``K1 = √γ |0⟩⟨1|``."""
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must lie in [0, 1]")
    return [
        np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=np.complex128),
        np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=np.complex128),
    ]


def phase_damping(lam: float) -> Kraus:
    """``K0 = diag(1, √(1−λ))``, ``K1 = diag(0, √λ)``; off-diagonals scale by ``√(1−λ)``."""
    if not 0 <= lam <= 1:
        raise ValueError("lambda must lie in [0, 1]")
    return [
        np.array([[1, 0], [0, np.sqrt(1 - lam)]], dtype=np.complex128),
        np.array([[0, 0], [0, np.sqrt(lam)]], dtype=np.complex128),
    ]


def pauli(px: float, py: float, pz: float) -> Kraus:
    pi = 1 - px - py - pz
    if min(px, py, pz, pi) < -_ATOL:
        raise ValueError("Pauli probabilities must be non-negative and sum to at most 1")
    return [
        np.sqrt(p) * m for p, m in zip((pi, px, py, pz), (_I, _X, _Y, _Z), strict=True) if p > 0
    ]


def bit_flip(p: float) -> Kraus:
    return pauli(p, 0.0, 0.0)


def phase_flip(p: float) -> Kraus:
    return pauli(0.0, 0.0, p)


def coherent_overrotation(axis: str, epsilon: float) -> Kraus:
    """Unitary ``exp(−i ε P/2)`` — a systematic calibration error, single Kraus operator."""
    p = PAULIS[axis.lower()]
    return [np.cos(epsilon / 2) * _I - 1j * np.sin(epsilon / 2) * p]


def thermal_relaxation(
    t1_us: float, t2_us: float, duration_us: float, excited_population: float = 0.0
) -> Kraus:
    """T1/T2 relaxation over ``duration_us``; requires ``T2 ≤ 2 T1``.

    Built from its action on the Bloch components (see module docstring) via the Choi matrix, so
    the ``T1 < T2 ≤ 2T1`` regime (non-Pauli) is handled exactly.
    """
    if t1_us <= 0 or t2_us <= 0:
        raise ValueError("T1 and T2 must be positive")
    if t2_us > 2 * t1_us + 1e-9:
        raise ValueError("T2 must not exceed 2·T1 (complete positivity)")
    if not 0 <= excited_population <= 1:
        raise ValueError("excited_population must lie in [0, 1]")
    e1 = np.exp(-duration_us / t1_us)
    e2 = np.exp(-duration_us / t2_us)
    n = excited_population

    def e(rho: Complex) -> Complex:
        tr = np.trace(rho)
        p1 = rho[1, 1] * e1 + n * (1 - e1) * tr
        return np.array([[tr - p1, rho[0, 1] * e2], [rho[1, 0] * e2, p1]], dtype=np.complex128)

    c = np.zeros((4, 4), dtype=np.complex128)
    for i in range(2):
        for j in range(2):
            basis = np.zeros((2, 2), dtype=np.complex128)
            basis[i, j] = 1
            c += np.kron(basis, e(basis))
    return choi_to_kraus(c)


# --- classical readout error ----------------------------------------------------------------------


@dataclass(frozen=True)
class ReadoutError:
    """Independent per-qubit assignment errors applied to a probability vector.

    ``p01 = P(read 1 | true 0)``, ``p10 = P(read 0 | true 1)``; the single-qubit confusion
    matrix ``M[read, true]`` is ``[[1−p01, p10], [p01, 1−p10]]``.  A list gives per-qubit
    values (qubit 0 first = most significant bit).  Readout noise is a classical stochastic map
    on the diagonal, so it is applied after simulation rather than as a Kraus channel.
    """

    p01: float | tuple[float, ...] = 0.0
    p10: float | tuple[float, ...] = 0.0

    def __post_init__(self) -> None:
        for name, v in (("p01", self.p01), ("p10", self.p10)):
            vals = v if isinstance(v, tuple) else (v,)
            if any(not 0 <= x <= 1 for x in vals):
                raise ValueError(f"{name} must lie in [0, 1], got {v}")
        if isinstance(self.p01, tuple) and isinstance(self.p10, tuple):
            if len(self.p01) != len(self.p10):
                raise ValueError("per-qubit p01 and p10 must have the same length")

    @classmethod
    def per_qubit(cls, pairs: Sequence[tuple[float, float]]) -> ReadoutError:
        return cls(tuple(a for a, _ in pairs), tuple(b for _, b in pairs))

    def _pair(self, q: int, n: int) -> tuple[float, float]:
        for v in (self.p01, self.p10):
            if isinstance(v, tuple) and len(v) != n:
                raise ValueError(f"per-qubit readout error has {len(v)} entries, circuit has {n}")
        a = self.p01[q] if isinstance(self.p01, tuple) else self.p01
        b = self.p10[q] if isinstance(self.p10, tuple) else self.p10
        return a, b

    @property
    def confusion(self) -> Real:
        a, b = self._pair(0, 1)
        return np.array([[1 - a, b], [a, 1 - b]])

    def confusion_for(self, q: int, n: int) -> Real:
        a, b = self._pair(q, n)
        return np.array([[1 - a, b], [a, 1 - b]])

    def apply(self, p: Real, n_qubits: int) -> Real:
        """``p_read = (M_0 ⊗ … ⊗ M_{n−1}) p`` applied axis by axis."""
        t = np.asarray(p, dtype=np.float64).reshape((2,) * n_qubits)
        for q in range(n_qubits):
            t = np.moveaxis(np.tensordot(self.confusion_for(q, n_qubits), t, axes=([1], [q])), 0, q)
        return t.reshape(-1)

    def is_trivial(self) -> bool:
        return not np.any(self.p01) and not np.any(self.p10)
