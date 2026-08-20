"""``Backend`` adapter over the orkan C density-matrix simulator.

The low-level ctypes binding lives in :mod:`ptqnn.backends._orkan_ffi`; this module is the thin
circuit-aware layer on top of it.  It allocates one ``MIXED_TILED`` state per
:meth:`OrkanDensityMatrixBackend.probabilities` call, walks ``circuit.ops``, and reads the
computational-basis distribution back out of the tiled buffer.  Results must agree with
:class:`ptqnn.backends.reference.DensityMatrixBackend` to machine precision — that is what
``tests/test_orkan_backend.py`` asserts.

Index conventions
-----------------
Our qubit *labels* pass through unchanged: circuit qubit ``q`` is orkan qubit ``q``.  Only the
*basis index* differs — orkan is lsb0, we are msb0 — and that is a bit reversal applied once, at
readout (:func:`ptqnn.backends._orkan_ffi.msb0_permutation`).  ``apply_2q_unitary(q0, q1, U)``
already takes ``U`` indexed ``2·b_q0 + b_q1``, i.e. exactly the IR's "first listed qubit = most
significant factor", so ``crx(c, t, θ)`` matrices are passed through verbatim.

orkan's ``rx``/``ry``/``rz`` are ``exp(-iθP/2)``, the same convention as
:func:`ptqnn.circuits.ir.rx` (asserted in ``test_orkan_rx_convention_matches_ours``); every gate
routed to a native kernel additionally has its matrix compared against the IR's before dispatch,
so a mislabelled :class:`~ptqnn.circuits.ir.Gate` falls back to the generic
``single_from_mat``/``two_from_mat`` path rather than silently applying the wrong operator.

Two-qubit channels
------------------
orkan has no 2-qubit ``channel_2q``; ``channel_1q`` is all there is.  Two routes are implemented:

* **mixture of unitaries** (the default, and what depolarising/Pauli channels are): if every
  ``K_i = √p_i U_i`` with ``U_i`` unitary — detected numerically from ``K_i† K_i = p_i I`` — then
  ``ρ ↦ Σ_i p_i U_i ρ U_i†`` is evaluated with a scratch copy of orkan's *storage buffer*.  orkan
  exports ``state_cp`` but no axpy, so the accumulation is done in numpy directly on the buffer.
  This is layout-agnostic and exact: the tiled buffer is a linear indexing of the matrix entries
  (padding excluded, and it stays zero), so a real-coefficient linear combination of buffers is
  the same linear combination of density matrices.  Cost: ``len(kraus)`` two-qubit gates plus
  three passes over a buffer of ``≈ 2^(2n-1)`` complex numbers.
* **general Kraus set** (amplitude damping on two qubits, and anything else that is not a
  mixture of unitaries): the density matrix is pulled out with ``to_numpy``, the channel is
  applied by :func:`ptqnn.backends.reference.apply_channel`, and the result is written back with
  :func:`from_numpy`.  This materialises the full ``2^n × 2^n`` matrix and is *slow*; the
  first one warns (once per backend instance) and all are counted in
  ``stats["slow_path_channels"]``.

Threading
---------
orkan parallelises with OpenMP and reads ``OMP_NUM_THREADS`` at load time (``orkan/docs/API.md``
§"Thread count controlled via OMP_NUM_THREADS environment variable").  The ffi survey on this
64-core box, replayed at 11 qubits (33 native 1q gates + 30 ``two_from_mat`` + 33 1q Kraus per
repetition, 3 repetitions):

===================  ========
``OMP_NUM_THREADS``  time (s)
===================  ========
1                    2.40
8                    0.41
16                   0.23
32                   0.25
64                   0.90
===================  ========

so **32 threads is optimal and 64 is ~3.6× slower than 32** — oversubscribing the SMT siblings
costs far more than it buys on this memory-bound kernel.  Hence the default ``n_threads=32``.
The variable is only honoured if it is set *before* ``liborkan.so`` is dlopen'd, so
:class:`OrkanDensityMatrixBackend` sets it in ``__init__`` (never overriding a value the user
already exported) and constructs the first state afterwards.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from ptqnn.backends import _orkan_ffi as ffi
from ptqnn.backends import reference
from ptqnn.circuits import ir
from ptqnn.circuits.ir import Channel, Circuit, Gate

__all__ = ["OrkanDensityMatrixBackend", "diagonal", "from_numpy"]

Complex = npt.NDArray[np.complex128]
Real = npt.NDArray[np.float64]

#: Tolerance for "is this Kraus set a mixture of unitaries" and for gate-matrix identification.
_ATOL = 1e-12

_ONE_Q: dict[str, Callable[[tuple[float, ...]], Complex]] = {
    "h": lambda p: ir.h(0).matrix,
    "x": lambda p: ir.x(0).matrix,
    "y": lambda p: ir.y(0).matrix,
    "z": lambda p: ir.z(0).matrix,
    "rx": lambda p: ir.rx(0, p[0]).matrix,
    "ry": lambda p: ir.ry(0, p[0]).matrix,
    "rz": lambda p: ir.rz(0, p[0]).matrix,
}
_CX4 = ir.cx(0, 1).matrix


# ---------------------------------------------------------------------------
# Tiled-buffer helpers
#
# Layout (``_orkan_ffi`` module docstring, ``orkan/src/state/state_tiled.c``): lower-triangular
# at tile level, row-major within a tile,
#     idx(r, c) = (tr*(tr+1)/2 + tc)*TILE_DIM² + lr*TILE_DIM + lc,   tr = r >> log2 TILE_DIM …
# valid whenever ``tr >= tc``.  Diagonal tiles store *both* halves (that is the point of the
# layout: "for within-tile operations no conjugation is needed because all 4 butterfly elements
# are directly accessible within the tile"), so a dense write must fill them both.  For
# ``dim < TILE_DIM`` the single tile is padded; the padding stays zero.
# ---------------------------------------------------------------------------


def _tiled_index(
    rows: npt.NDArray[np.int64], cols: npt.NDArray[np.int64], tile: int
) -> npt.NDArray[np.int64]:
    """Flat storage index of element ``(r, c)``; requires ``r >> log ≥ c >> log``."""
    log = tile.bit_length() - 1
    tr = rows >> log
    tc = cols >> log
    idx: npt.NDArray[np.int64] = (
        (tr * (tr + 1) // 2 + tc) * (tile * tile) + (rows & (tile - 1)) * tile + (cols & (tile - 1))
    )
    return idx


def _row_step(dim: int) -> int:
    """Rows per pass, capping the temporary index arrays at ~64 MB (mirrors ``_orkan_ffi``)."""
    return max(1, min(dim, (1 << 23) // max(1, dim)))


def _require_tiled(state: ffi.OrkanState) -> tuple[Complex, int, int]:
    if state.kind is not ffi.StateType.MIXED_TILED:
        raise ffi.OrkanTypeError(f"expected a MIXED_TILED state, got {state.kind.name}")
    return state._raw(), 1 << state.n_qubits, ffi.load_orkan().tile_dim


def from_numpy(state: ffi.OrkanState, rho: Complex, order: str = "msb0") -> None:
    """Overwrite a ``MIXED_TILED`` state with the dense density matrix ``rho``.

    The inverse of :meth:`ptqnn.backends._orkan_ffi.OrkanState.to_numpy`; ``order`` has the same
    meaning (``"msb0"`` applies the bit-reversal permutation on both axes on the way in).
    """
    raw, dim, tile = _require_tiled(state)
    m = np.asarray(rho, dtype=np.complex128)
    if m.shape != (dim, dim):
        raise ValueError(f"expected a {dim}x{dim} density matrix, got shape {m.shape}")
    if order not in ("lsb0", "msb0"):
        raise ValueError(f"order must be 'lsb0' or 'msb0', got {order!r}")
    if order == "msb0":
        perm = ffi.msb0_permutation(state.n_qubits)
        m = m[np.ix_(perm, perm)]
    raw[:] = 0.0  # padding slots of a sub-tile-sized matrix are never written below
    log = tile.bit_length() - 1
    step = _row_step(dim)
    cols = np.arange(dim, dtype=np.int64)[None, :]
    for start in range(0, dim, step):
        stop = min(start + step, dim)
        rows = np.arange(start, stop, dtype=np.int64)[:, None]
        keep = (rows >> log) >= (cols >> log)
        r, c = np.broadcast_arrays(rows, cols)
        raw[_tiled_index(r[keep], c[keep], tile)] = m[start:stop][keep]


def load_pure(state: ffi.OrkanState, psi: Complex, order: str = "msb0") -> None:
    """Overwrite a ``MIXED_TILED`` state with ``|ψ⟩⟨ψ|``, one row block at a time.

    Cheaper than ``from_numpy(np.outer(psi, psi.conj()))``: the outer product is never
    materialised in full.
    """
    raw, dim, tile = _require_tiled(state)
    v = np.asarray(psi, dtype=np.complex128).ravel()
    if v.shape != (dim,):
        raise ValueError(f"expected a length-{dim} statevector, got shape {v.shape}")
    if order not in ("lsb0", "msb0"):
        raise ValueError(f"order must be 'lsb0' or 'msb0', got {order!r}")
    if order == "msb0":
        v = v[ffi.msb0_permutation(state.n_qubits)]
    raw[:] = 0.0
    log = tile.bit_length() - 1
    step = _row_step(dim)
    cols = np.arange(dim, dtype=np.int64)[None, :]
    vc = v.conj()
    for start in range(0, dim, step):
        stop = min(start + step, dim)
        rows = np.arange(start, stop, dtype=np.int64)[:, None]
        keep = (rows >> log) >= (cols >> log)
        r, c = np.broadcast_arrays(rows, cols)
        raw[_tiled_index(r[keep], c[keep], tile)] = np.outer(v[start:stop], vc)[keep]


def diagonal(state: ffi.OrkanState, order: str = "msb0") -> Real:
    """Real diagonal of a ``MIXED_TILED`` state, gathered straight from the tiled buffer.

    ``2^n`` gathers instead of the ``2^{2n}`` reconstruction ``to_numpy`` would do.
    """
    raw, dim, tile = _require_tiled(state)
    if order not in ("lsb0", "msb0"):
        raise ValueError(f"order must be 'lsb0' or 'msb0', got {order!r}")
    i = np.arange(dim, dtype=np.int64)
    d = raw[_tiled_index(i, i, tile)].real.copy()
    if order == "msb0":
        d = d[ffi.msb0_permutation(state.n_qubits)]
    return np.asarray(d, dtype=np.float64)


# ---------------------------------------------------------------------------
# Kraus analysis
# ---------------------------------------------------------------------------


def unitary_mixture(kraus: Sequence[Complex], atol: float = _ATOL) -> list[tuple[float, Complex]]:
    """``[(p_i, U_i)]`` with ``K_i = √p_i U_i`` and ``U_i`` unitary, or ``[]`` if not of that form.

    ``K† K = p I`` characterises it; ``p = ‖K‖_F² / d``.  True for depolarising and Pauli
    channels (every Kraus operator is a scaled Pauli string), false for amplitude damping.
    """
    out: list[tuple[float, Complex]] = []
    for k in kraus:
        d = k.shape[0]
        p = float(np.vdot(k, k).real / d)
        if p <= atol:
            continue  # a zero Kraus operator contributes nothing
        if not np.allclose(k.conj().T @ k, p * np.eye(d), atol=atol):
            return []
        out.append((p, k / np.sqrt(p)))
    return out


# ---------------------------------------------------------------------------
# The backend
# ---------------------------------------------------------------------------


class OrkanDensityMatrixBackend:
    """Exact mixed-state evolution on orkan's ``MIXED_TILED`` C kernels.

    Same contract as :class:`ptqnn.backends.reference.DensityMatrixBackend`, ~2 orders of
    magnitude faster at 11 qubits.  Memory is ``≈ 16·2^{2n-1}`` bytes (34 MB at n = 11, 2 GB at
    n = 14, orkan's practical ceiling).

    Parameters
    ----------
    n_threads:
        Value for ``OMP_NUM_THREADS``, exported here unless the environment already sets it.
        The default 32 is the survey optimum on this 64-core box (see the module docstring: 64
        threads is ~3.6× slower than 32).  ``None`` leaves the environment alone.
    mixture_fast_path:
        Apply 2-qubit mixture-of-unitary channels with in-buffer accumulation instead of the
        dense numpy round trip.  Only turn it off to exercise the slow path in tests.
    native_gates:
        Route ``h/x/y/z/rx/ry/rz/cx`` to orkan's specialised kernels.  Only turn it off to check
        the generic ``single_from_mat``/``two_from_mat`` path against them.
    """

    def __init__(
        self,
        n_threads: int | None = 32,
        *,
        mixture_fast_path: bool = True,
        native_gates: bool = True,
    ) -> None:
        if n_threads is not None and not os.environ.get("OMP_NUM_THREADS"):
            # Must precede the dlopen; libgomp reads the variable when it initialises.
            os.environ["OMP_NUM_THREADS"] = str(int(n_threads))
        self.n_threads = n_threads
        self.mixture_fast_path = mixture_fast_path
        self.native_gates = native_gates
        self.stats: dict[str, int] = {
            "circuits": 0,
            "two_qubit_channels": 0,
            "slow_path_channels": 0,
        }
        self._warned_slow_path = False

    @property
    def name(self) -> str:
        return "orkan-density-matrix"

    # -- public API ---------------------------------------------------------

    def probabilities(self, circuit: Circuit) -> Real:
        state = self._run(circuit)
        try:
            return diagonal(state, order="msb0")
        finally:
            state.free()

    def density_matrix(self, circuit: Circuit) -> Complex:
        """Full ``2^n × 2^n`` density matrix in msb0 order — for tests and Choi-level checks."""
        state = self._run(circuit)
        try:
            return state.to_numpy(order="msb0")
        finally:
            state.free()

    # -- evolution ----------------------------------------------------------

    def _run(self, circuit: Circuit) -> ffi.OrkanState:
        """Evolve a fresh ``MIXED_TILED`` state; the caller owns (and must free) the result."""
        n = circuit.n_qubits
        if n < 1:
            raise ValueError("orkan needs at least one qubit")
        state = (
            ffi.OrkanState.zero(n, ffi.StateType.MIXED_TILED)
            if circuit.initial_state is None
            else ffi.OrkanState(n, ffi.StateType.MIXED_TILED)
        )
        try:
            if circuit.initial_state is not None:
                load_pure(state, circuit.initial_state, order="msb0")
            for op in circuit.ops:
                self._apply(state, op)
        except BaseException:
            state.free()
            raise
        self.stats["circuits"] += 1
        return state

    def _apply(self, state: ffi.OrkanState, op: ir.Op) -> None:
        if isinstance(op, Gate):
            self._apply_gate(state, op)
        else:
            self._apply_channel(state, op)

    def _apply_gate(self, state: ffi.OrkanState, gate: Gate) -> None:
        if gate.n_qubits == 1:
            if not (self.native_gates and self._native_1q(state, gate)):
                state.apply_1q_unitary(gate.qubits[0], gate.matrix)
            return
        if gate.n_qubits == 2:
            q0, q1 = gate.qubits
            if self.native_gates and gate.name == "cx" and _is(gate.matrix, _CX4):
                state.cx(q0, q1)
            else:
                # `u` is indexed 2·b_q0 + b_q1 — the IR's own multi-qubit convention.
                state.apply_2q_unitary(q0, q1, gate.matrix)
            return
        raise NotImplementedError(
            f"orkan has no arbitrary {gate.n_qubits}-qubit unitary (gate {gate.name!r}); "
            "decompose it into 1- and 2-qubit gates first"
        )

    @staticmethod
    def _native_1q(state: ffi.OrkanState, gate: Gate) -> bool:
        """Dispatch to an orkan kernel iff the gate's matrix really is the one its name claims."""
        build = _ONE_Q.get(gate.name)
        if build is None or len(gate.params) != (1 if gate.name.startswith("r") else 0):
            return False
        if not _is(build(gate.params), gate.matrix):
            return False
        getattr(state, gate.name)(gate.qubits[0], *gate.params)
        return True

    def _apply_channel(self, state: ffi.OrkanState, chan: Channel) -> None:
        if chan.n_qubits == 1:
            state.apply_kraus_1q(chan.qubits[0], chan.kraus)
            return
        if chan.n_qubits != 2:
            raise NotImplementedError(
                f"orkan has no {chan.n_qubits}-qubit channel (channel {chan.name!r})"
            )
        self.stats["two_qubit_channels"] += 1
        mixture = unitary_mixture(chan.kraus) if self.mixture_fast_path else []
        if mixture:
            self._apply_unitary_mixture(state, chan.qubits, mixture)
        else:
            self._apply_channel_dense(state, chan)

    @staticmethod
    def _apply_unitary_mixture(
        state: ffi.OrkanState, qubits: tuple[int, ...], mixture: list[tuple[float, Complex]]
    ) -> None:
        """``ρ ↦ Σ_i p_i U_i ρ U_i†`` accumulated in orkan's own storage buffer.

        Valid because the tiled buffer is a *linear* map of the matrix entries: writing
        ``Σ_i p_i buf(U_i ρ U_i†)`` into it is the same as writing ``buf(Σ_i p_i U_i ρ U_i†)``.
        (Real coefficients matter: any conjugate-mirrored slot stays consistent too.)
        """
        raw, _, _ = _require_tiled(state)
        q0, q1 = qubits
        base = raw.copy()
        acc = np.zeros_like(raw)
        for i, (p, u) in enumerate(mixture):
            if i:
                raw[:] = base
            state.apply_2q_unitary(q0, q1, u)
            acc += p * raw
        raw[:] = acc

    def _apply_channel_dense(self, state: ffi.OrkanState, chan: Channel) -> None:
        """Dense numpy round trip — the only route for a 2-qubit non-unitary-mixture channel."""
        self.stats["slow_path_channels"] += 1
        if not self._warned_slow_path:
            self._warned_slow_path = True
            warnings.warn(
                f"orkan has no 2-qubit channel kernel and {chan.name!r} is not a mixture of "
                "unitaries, so it takes the slow path: the full 2^n x 2^n density matrix is "
                "pulled into numpy, the channel applied there, and the matrix written back. "
                "Expect this to dominate the runtime.",
                RuntimeWarning,
                stacklevel=2,
            )
        rho = state.to_numpy(order="msb0")
        rho = reference.apply_channel(rho, chan, state.n_qubits)
        from_numpy(state, rho, order="msb0")


def _is(a: Complex, b: Complex) -> bool:
    return bool(np.allclose(a, b, atol=_ATOL))
