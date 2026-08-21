"""Batched exact statevector evolution — the one kernel behind every task's "fast path".

Review 002 finding 7: Flappy, Iris and ECG had each grown a private ``rx``/``crx``-only
statevector loop (basis-column trick, output-qubit contraction, batched windows).  They are the
same computation: a *batch* of pure states ``(B, 2^n)`` pushed through the same unitary circuit,
one pass per gate instead of one per gate and state.  This module is that computation, for any
gate the IR can express (generic ``2^k × 2^k`` unitaries on 1 or 2 qubits), pinned to the
numpy reference on full distributions in ``tests/test_batched_backend.py``.

It is unitary-only on purpose: a :class:`~ptqnn.circuits.ir.Channel` raises ``TypeError``.  A
noisy evaluation goes through a density-matrix backend (or trajectories); there is no silent
slow path here.

Conventions (``ptqnn.circuits.ir``): qubit 0 is the most significant bit of the basis index; a
2-qubit gate's matrix is indexed ``2·b_q0 + b_q1`` in the order of ``gate.qubits``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ptqnn.circuits import ir

Complex = npt.NDArray[np.complex128]
Real = npt.NDArray[np.float64]


def apply_gate_batched(state: Complex, gate: ir.Gate, n: int) -> Complex:
    """``state[b] ← U_gate state[b]`` for every row of the ``(B, 2^n)`` batch (returns a new
    array; the input is not modified).

    The batch axis is carried as a leading tensor axis and the gate is contracted into the
    target qubit axes with one ``tensordot`` — the same primitive as
    :func:`ptqnn.backends.reference._apply_matrix_to_axes`, with the batch in front.
    """
    B = state.shape[0]
    t = state.reshape((B,) + (2,) * n)
    k = gate.n_qubits
    axes = [1 + q for q in gate.qubits]  # shift past the batch axis
    mk = gate.matrix.reshape((2,) * (2 * k))
    out = np.tensordot(mk, t, axes=(list(range(k, 2 * k)), axes))
    # tensordot leaves the gate's row axes first; the batch axis is now at position k.
    out = np.moveaxis(out, list(range(k)), axes)
    return np.ascontiguousarray(out.reshape(B, 2**n))


def run_batch(circuit: ir.Circuit, states: npt.ArrayLike) -> Complex:
    """Evolve every row of ``states`` (``(B, 2^n)``, complex) through ``circuit``'s gates.

    ``circuit.initial_state`` is ignored: the rows *are* the initial states.  Raises
    ``TypeError`` on the first :class:`~ptqnn.circuits.ir.Channel` — the batched path is exact
    for pure states only.
    """
    n = circuit.n_qubits
    psi = np.array(states, dtype=np.complex128, copy=True)
    if psi.ndim != 2 or psi.shape[1] != 2**n:
        raise ValueError(f"states must be (B, {2**n}) for {n} qubits, got {psi.shape}")
    for op in circuit.ops:
        if not isinstance(op, ir.Gate):
            raise TypeError(
                f"batched statevector path is unitary-only; got channel {op.name!r} on "
                f"{op.qubits} — use a density-matrix or trajectory backend"
            )
        psi = apply_gate_batched(psi, op, n)
    return psi


def batch_probabilities(circuit: ir.Circuit, states: npt.ArrayLike) -> Real:
    """``|ψ_b|²`` for every row after :func:`run_batch` — ``(B, 2^n)``, qubit 0 = MSB."""
    out = run_batch(circuit, states)
    return np.asarray(np.abs(out) ** 2, dtype=np.float64)


def basis_columns(circuit: ir.Circuit, qubits: Sequence[int]) -> Complex:
    """``V[:, b] = U_circuit |b on qubits, 0 elsewhere⟩`` for all ``2^len(qubits)`` strings ``b``
    (``b``'s first bit goes to ``qubits[0]``).

    This is the "linear input" trick: when a circuit is *prepare inputs, then a fixed unitary*,
    the network needs to be evolved once per parameter vector and every later input costs one
    ``V @ amplitudes`` product.  Valid for pure states only (the noisy analogue is the transfer
    matrix of ``experiments/flappy_noise.py``).
    """
    n = circuit.n_qubits
    qs = [int(q) for q in qubits]
    m = len(qs)
    states = np.zeros((2**m, 2**n), dtype=np.complex128)
    for b in range(2**m):
        idx = 0
        for j, q in enumerate(qs):
            if (b >> (m - 1 - j)) & 1:
                idx |= 1 << (n - 1 - q)
        states[b, idx] = 1.0
    return np.ascontiguousarray(run_batch(circuit, states).T)


def product_input_states(amplitudes: Complex, n_qubits: int, n_inputs: int) -> Complex:
    """Batch of product states ``⊗_i (α_i|0⟩ + β_i|1⟩)`` on the first ``n_inputs`` qubits and
    ``|0…0⟩`` on the rest.  ``amplitudes`` is ``(B, n_inputs, 2)``."""
    amps = np.asarray(amplitudes, dtype=np.complex128)
    if amps.ndim != 3 or amps.shape[1:] != (n_inputs, 2):
        raise ValueError(f"amplitudes must be (B, {n_inputs}, 2), got {amps.shape}")
    B = amps.shape[0]
    v = np.ones((B, 1), dtype=np.complex128)
    for i in range(n_inputs):
        v = (v[:, :, None] * amps[:, i, None, :]).reshape(B, -1)
    state = np.zeros((B, 2**n_qubits), dtype=np.complex128)
    state[:, :: 2 ** (n_qubits - n_inputs)] = v  # the trailing qubits are |0⟩ (low bits)
    return state


class BatchedStatevectorBackend:
    """The :class:`~ptqnn.backends.base.Backend` face of this module (B = 1), plus
    :meth:`batch_probabilities` for callers that have many initial states per circuit."""

    name = "numpy-statevector-batched"

    def probabilities(self, circuit: ir.Circuit) -> Real:
        n = circuit.n_qubits
        if circuit.initial_state is not None:
            psi0 = np.asarray(circuit.initial_state, dtype=np.complex128).reshape(1, -1)
        else:
            psi0 = np.zeros((1, 2**n), dtype=np.complex128)
            psi0[0, 0] = 1.0
        return np.asarray(batch_probabilities(circuit, psi0)[0], dtype=np.float64)

    def batch_probabilities(self, circuit: ir.Circuit, states: npt.ArrayLike) -> Real:
        return batch_probabilities(circuit, states)
