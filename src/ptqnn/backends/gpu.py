"""CuPy (GPU) versions of the reference density-matrix and trajectory backends.

The reference implementation is written against the NumPy API only (``tensordot``,
``moveaxis``, ``reshape``), and CuPy implements the ``__array_function__`` protocol, so the very
same functions run on the GPU once the state *and* the gate matrices live in device memory.
This module only moves data and dispatches; the numerics are the ones tested in
``tests/test_reference_backend.py`` and ``tests/test_trajectories.py``.

On this machine (WSL2) CuPy needs ``LD_LIBRARY_PATH=/usr/lib/wsl/lib`` to find ``libcuda``; see
``CLAUDE.md`` gotchas.  Import failures are turned into :class:`GpuUnavailable` at construction
time so the module itself always imports.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from ptqnn.backends import reference as ref
from ptqnn.backends import trajectories as traj
from ptqnn.circuits import ir

Real = npt.NDArray[np.float64]


class GpuUnavailable(RuntimeError):
    pass


def _cupy() -> Any:
    try:
        import cupy as cp

        cp.cuda.runtime.getDeviceCount()
    except Exception as e:  # pragma: no cover - depends on the machine
        raise GpuUnavailable(f"CuPy/CUDA not usable: {e}") from e
    return cp


def gpu_available() -> bool:
    try:
        _cupy()
    except GpuUnavailable:
        return False
    return True


def _to_device(cp: Any, circuit: ir.Circuit) -> ir.Circuit:
    """A copy of the circuit whose matrices are CuPy arrays (witness checks skipped)."""
    out = ir.Circuit.__new__(ir.Circuit)
    out.n_qubits = circuit.n_qubits
    out.initial_state = None if circuit.initial_state is None else cp.asarray(circuit.initial_state)
    ops: list[ir.Op] = []
    for op in circuit.ops:
        if isinstance(op, ir.Gate):
            g = ir.Gate.__new__(ir.Gate)
            object.__setattr__(g, "qubits", op.qubits)
            object.__setattr__(g, "matrix", cp.asarray(op.matrix))
            object.__setattr__(g, "name", op.name)
            object.__setattr__(g, "params", op.params)
            ops.append(g)
        else:
            c = ir.Channel.__new__(ir.Channel)
            object.__setattr__(c, "qubits", op.qubits)
            object.__setattr__(c, "kraus", [cp.asarray(k) for k in op.kraus])
            object.__setattr__(c, "name", op.name)
            ops.append(c)
    out.ops = ops
    return out


class GpuDensityMatrixBackend:
    """Exact mixed-state evolution on the GPU; 13–14 qubits fit in 24 GB."""

    name = "cupy-density-matrix"

    def __init__(self) -> None:
        self.cp = _cupy()

    def probabilities(self, circuit: ir.Circuit) -> Real:
        cp = self.cp
        dev = _to_device(cp, circuit)
        n = circuit.n_qubits
        if dev.initial_state is not None:
            rho = cp.outer(dev.initial_state, dev.initial_state.conj())
        else:
            rho = cp.zeros((2**n, 2**n), dtype=cp.complex128)
            rho[0, 0] = 1.0
        for op in dev.ops:
            rho = ref.apply_op_dm(rho, op, n)
        return np.asarray(cp.asnumpy(cp.real(cp.diagonal(rho))), dtype=np.float64)


class GpuTrajectoryBackend:
    """Batched Monte-Carlo trajectories on the GPU (same sampling scheme as the CPU version,
    but the random branch choice uses a CuPy generator, so streams differ from the CPU one)."""

    def __init__(self, n_trajectories: int = 1000, seed: int = 0, batch_size: int = 1024):
        self.cp = _cupy()
        self.n_trajectories = n_trajectories
        self.seed = seed
        self.batch_size = batch_size

    @property
    def name(self) -> str:
        return f"cupy-trajectories(n={self.n_trajectories}, seed={self.seed})"

    def probabilities(self, circuit: ir.Circuit) -> Real:
        cp = self.cp
        n = circuit.n_qubits
        dev = _to_device(cp, circuit)
        rng = cp.random.default_rng(self.seed)
        if dev.initial_state is not None:
            psi0 = dev.initial_state
        else:
            psi0 = cp.zeros(2**n, dtype=cp.complex128)
            psi0[0] = 1.0
        total = 1 if circuit.is_unitary() else self.n_trajectories
        acc = cp.zeros(2**n, dtype=cp.float64)
        done = 0
        while done < total:
            b = min(self.batch_size, total - done)
            batch = cp.tile(psi0, (b, 1))
            for op in dev.ops:
                if isinstance(op, ir.Gate):
                    batch = traj._apply_matrix_batched(batch, op.matrix, op.qubits, n)
                else:
                    batch = traj._apply_channel_batched(batch, op, n, rng)
            acc += cp.sum(cp.abs(batch) ** 2, axis=0)
            done += b
        return np.asarray(cp.asnumpy(acc / total), dtype=np.float64)
