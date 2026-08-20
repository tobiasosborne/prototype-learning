"""Monte-Carlo Kraus unravelling on batched statevectors — the noisy backend beyond ~12 qubits.

A density-matrix evolution ``ρ ↦ Σ_i K_i ρ K_i†`` equals the average over pure-state
trajectories in which, at each channel, branch ``i`` is chosen with probability
``p_i = ‖K_i ψ‖²`` and the state becomes ``K_i ψ / √p_i`` (quantum-jump / Monte-Carlo wave
function method; standard, see e.g. Dalibard–Castin–Mølmer PRL 68, 580 (1992)).  The estimated
outcome distribution is the mean of ``|ψ|²`` over ``n_trajectories``, with statistical error
``√(p(1−p)/N)`` per outcome — tests assert agreement with the exact density-matrix backend at
four sigma.

Trajectories are processed in batches of shape ``(B, 2^n)`` so every gate is one ``tensordot``;
memory is ``B · 2^n · 16`` bytes (Iris at 17 qubits, ``B = 256``: 0.5 GB).  Unitary circuits
are exact with a single trajectory.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ptqnn.backends import reference as ref
from ptqnn.circuits import ir

Complex = npt.NDArray[np.complex128]
Real = npt.NDArray[np.float64]


def _apply_matrix_batched(batch: Complex, m: Complex, qubits: Sequence[int], n: int) -> Complex:
    """``m`` on ``qubits`` of every trajectory in ``batch`` (shape ``(B, 2^n)``)."""
    b = batch.shape[0]
    t = batch.reshape((b, *([2] * n)))
    axes = [q + 1 for q in qubits]
    out = ref._apply_matrix_to_axes(t, m, axes, n + 1)
    return out.reshape(b, 2**n)


def _apply_channel_batched(
    batch: Complex, ch: ir.Channel, n: int, rng: np.random.Generator
) -> Complex:
    """Sample one Kraus branch per trajectory and renormalise."""
    b = batch.shape[0]
    if len(ch.kraus) == 1:
        return _apply_matrix_batched(batch, ch.kraus[0], ch.qubits, n)
    branches = [_apply_matrix_batched(batch, k, ch.qubits, n) for k in ch.kraus]
    norms = np.stack([np.sum(np.abs(x) ** 2, axis=1) for x in branches], axis=1)  # (B, m)
    cdf = np.cumsum(norms, axis=1)
    u = rng.random(b) * cdf[:, -1]
    choice = np.minimum((u[:, None] >= cdf).sum(axis=1), len(branches) - 1)
    out = np.empty_like(batch)
    for i, br in enumerate(branches):
        sel = choice == i
        if np.any(sel):
            out[sel] = br[sel] / np.sqrt(norms[sel, i])[:, None]
    return out


class TrajectoryBackend:
    """``Backend`` estimating the noisy outcome distribution from ``n_trajectories`` samples."""

    def __init__(self, n_trajectories: int = 1000, seed: int = 0, batch_size: int = 256):
        if n_trajectories < 1 or int(n_trajectories) != n_trajectories:
            raise ValueError("n_trajectories must be a positive integer")
        if batch_size < 1 or int(batch_size) != batch_size:
            raise ValueError("batch_size must be a positive integer")
        self.n_trajectories = n_trajectories
        self.seed = seed
        self.batch_size = batch_size

    @property
    def name(self) -> str:
        return f"trajectories(n={self.n_trajectories}, seed={self.seed})"

    def probabilities(self, circuit: ir.Circuit) -> Real:
        n = circuit.n_qubits
        if circuit.is_unitary():
            return ref.StatevectorBackend().probabilities(circuit)
        rng = np.random.default_rng(self.seed)
        psi0 = circuit.initial_state if circuit.initial_state is not None else ref.zero_state(n)
        acc = np.zeros(2**n, dtype=np.float64)
        done = 0
        while done < self.n_trajectories:
            b = min(self.batch_size, self.n_trajectories - done)
            batch = np.tile(psi0, (b, 1))
            for op in circuit.ops:
                if isinstance(op, ir.Gate):
                    batch = _apply_matrix_batched(batch, op.matrix, op.qubits, n)
                else:
                    batch = _apply_channel_batched(batch, op, n, rng)
            acc += np.sum(np.abs(batch) ** 2, axis=0)
            done += b
        return acc / self.n_trajectories
