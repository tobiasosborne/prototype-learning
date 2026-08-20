"""Numpy reference backend — the oracle every faster backend must agree with.

Statevector and density-matrix simulation by tensor reshaping: an ``n``-qubit state is viewed as
an array of shape ``(2,)*n`` whose axis ``q`` is qubit ``q`` (so qubit 0 is the most significant
bit of the flat index, matching :mod:`ptqnn.circuits.ir`).  A ``k``-qubit operator is contracted
into the ``k`` corresponding axes with ``np.tensordot`` and the axes moved back.  Density
matrices use shape ``(2,)*2n``: axes ``0..n-1`` are the row (ket) qubits, ``n..2n-1`` the column
(bra) qubits, and a channel acts as ``ρ ↦ Σ_i K_i ρ K_i†``.

The code favours obviousness over speed; 12-qubit density matrices (4096²) are fine, larger
ones belong to the orkan or trajectory backends.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ptqnn.circuits.ir import Channel, Circuit, Gate, Op

Complex = npt.NDArray[np.complex128]
Real = npt.NDArray[np.float64]


def zero_state(n: int) -> Complex:
    psi = np.zeros(2**n, dtype=np.complex128)
    psi[0] = 1.0
    return psi


def zero_density_matrix(n: int) -> Complex:
    rho = np.zeros((2**n, 2**n), dtype=np.complex128)
    rho[0, 0] = 1.0
    return rho


def _apply_matrix_to_axes(
    tensor: Complex, m: Complex, axes: Sequence[int], n_axes_total: int
) -> Complex:
    """Contract ``m`` (shape ``2^k × 2^k``) into ``axes`` of ``tensor`` and restore axis order."""
    k = len(axes)
    mk = m.reshape((2,) * (2 * k))
    # tensordot contracts m's column axes (k..2k-1) with tensor's `axes`; the result has m's row
    # axes first, followed by the remaining tensor axes in order.  Move them back into place.
    out = np.tensordot(mk, tensor, axes=(list(range(k, 2 * k)), list(axes)))
    return np.moveaxis(out, list(range(k)), list(axes))


def apply_gate_sv(psi: Complex, gate: Gate, n: int) -> Complex:
    t = psi.reshape((2,) * n)
    t = _apply_matrix_to_axes(t, gate.matrix, gate.qubits, n)
    return t.reshape(2**n)


def superoperator(kraus: Sequence[Complex]) -> Complex:
    """``S = Σ_i K_i ⊗ K_i*`` acting on the row-major vectorisation ``vec(X)[a·d + b] = X[a, b]``
    of the channel's own ``d × d`` block, since ``vec(A X B) = (A ⊗ Bᵀ) vec(X)``."""
    d2 = kraus[0].shape[0] ** 2
    s: Complex = np.zeros((d2, d2), dtype=np.complex128)
    for k in kraus:
        s = s + np.kron(k, k.conj())
    return s


def _apply_superop_dm(rho: Complex, s: Complex, qubits: Sequence[int], n: int) -> Complex:
    """Apply a ``d² × d²`` superoperator to the ket *and* bra axes of ``qubits`` in one matmul.

    The channel's ket axes ``q`` and bra axes ``q + n`` are moved to the front, the tensor is
    viewed as ``(d², rest)`` and multiplied once — one pass over memory per channel regardless of
    the number of Kraus operators (``reviews`` P9 finding: per-Kraus ``tensordot`` was ~80× the
    memory-bound cost at 11 qubits)."""
    k = len(qubits)
    d = 2**k
    t = rho.reshape((2,) * (2 * n))
    src = [*qubits, *(q + n for q in qubits)]
    t = np.moveaxis(t, src, list(range(2 * k)))
    shape = t.shape
    out = s @ t.reshape(d * d, -1)
    out = out.reshape(shape)
    out = np.moveaxis(out, list(range(2 * k)), src)
    return out.reshape(2**n, 2**n)


def apply_gate_dm(rho: Complex, gate: Gate, n: int) -> Complex:
    """``ρ ↦ U ρ U†`` as the superoperator ``U ⊗ U*`` on the gate's block."""
    return _apply_superop_dm(rho, superoperator([gate.matrix]), gate.qubits, n)


def apply_channel(rho: Complex, ch: Channel, n: int) -> Complex:
    """``ρ ↦ Σ_i K_i ρ K_i†`` via the block superoperator (one matmul)."""
    return _apply_superop_dm(rho, superoperator(ch.kraus), ch.qubits, n)


def apply_op_dm(rho: Complex, op: Op, n: int) -> Complex:
    if isinstance(op, Gate):
        return apply_gate_dm(rho, op, n)
    return apply_channel(rho, op, n)


def run_statevector(circuit: Circuit, psi0: Complex | None = None) -> Complex:
    """Exact pure-state evolution; raises if the circuit contains a non-unitary channel."""
    n = circuit.n_qubits
    if psi0 is None:
        psi0 = circuit.initial_state
    psi = zero_state(n) if psi0 is None else np.asarray(psi0, dtype=np.complex128).copy()
    for op in circuit.ops:
        if not isinstance(op, Gate):
            raise ValueError(f"statevector backend cannot apply channel {op.name!r}")
        psi = apply_gate_sv(psi, op, n)
    return psi


def run_density_matrix(circuit: Circuit, rho0: Complex | None = None) -> Complex:
    n = circuit.n_qubits
    if rho0 is None and circuit.initial_state is not None:
        rho0 = np.outer(circuit.initial_state, circuit.initial_state.conj())
    rho = zero_density_matrix(n) if rho0 is None else np.asarray(rho0, dtype=np.complex128).copy()
    for op in circuit.ops:
        rho = apply_op_dm(rho, op, n)
    return rho


def probabilities(state: Complex) -> Real:
    """Computational-basis probabilities of a statevector or density matrix."""
    if state.ndim == 1:
        return np.abs(state) ** 2
    return np.real(np.diagonal(state)).copy()


def marginal(p: Real, keep: Sequence[int]) -> Real:
    """Marginal distribution over ``keep`` (in the listed order, first = most significant)."""
    n = round(np.log2(p.size))
    t = p.reshape((2,) * n)
    drop = tuple(q for q in range(n) if q not in keep)
    s = t.sum(axis=drop) if drop else t
    remaining = [q for q in range(n) if q in keep]
    order = [remaining.index(q) for q in keep]
    return np.transpose(s, order).reshape(2 ** len(keep))


def sample_counts(p: Real, shots: int, rng: np.random.Generator) -> npt.NDArray[np.int64]:
    """Multinomial shot counts — the only place randomness enters a noiseless run."""
    p = np.clip(p, 0, None)
    return rng.multinomial(shots, p / p.sum()).astype(np.int64)


class StatevectorBackend:
    """Exact, noiseless.  The default for reproduction runs."""

    name = "numpy-statevector"

    def probabilities(self, circuit: Circuit) -> Real:
        return probabilities(run_statevector(circuit))


class DensityMatrixBackend:
    """Exact mixed-state evolution; accepts channels.  ≤ 12 qubits in practice."""

    name = "numpy-density-matrix"

    def probabilities(self, circuit: Circuit) -> Real:
        return probabilities(run_density_matrix(circuit))
