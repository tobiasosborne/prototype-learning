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


def apply_gate_dm(rho: Complex, gate: Gate, n: int) -> Complex:
    """``ρ ↦ U ρ U†`` via the ket axes then the bra axes (with ``U*``)."""
    t = rho.reshape((2,) * (2 * n))
    t = _apply_matrix_to_axes(t, gate.matrix, gate.qubits, 2 * n)
    t = _apply_matrix_to_axes(t, gate.matrix.conj(), [q + n for q in gate.qubits], 2 * n)
    return t.reshape(2**n, 2**n)


def apply_channel(rho: Complex, ch: Channel, n: int) -> Complex:
    """``ρ ↦ Σ_i K_i ρ K_i†``."""
    t = rho.reshape((2,) * (2 * n))
    out = np.zeros_like(t)
    bra_axes = [q + n for q in ch.qubits]
    for k in ch.kraus:
        s = _apply_matrix_to_axes(t, k, ch.qubits, 2 * n)
        out += _apply_matrix_to_axes(s, k.conj(), bra_axes, 2 * n)
    return out.reshape(2**n, 2**n)


def apply_op_dm(rho: Complex, op: Op, n: int) -> Complex:
    if isinstance(op, Gate):
        return apply_gate_dm(rho, op, n)
    return apply_channel(rho, op, n)


def run_statevector(circuit: Circuit, psi0: Complex | None = None) -> Complex:
    """Exact pure-state evolution; raises if the circuit contains a non-unitary channel."""
    n = circuit.n_qubits
    psi = zero_state(n) if psi0 is None else np.asarray(psi0, dtype=np.complex128).copy()
    for op in circuit.ops:
        if not isinstance(op, Gate):
            raise ValueError(f"statevector backend cannot apply channel {op.name!r}")
        psi = apply_gate_sv(psi, op, n)
    return psi


def run_density_matrix(circuit: Circuit, rho0: Complex | None = None) -> Complex:
    n = circuit.n_qubits
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
