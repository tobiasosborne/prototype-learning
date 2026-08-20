"""Low-level ctypes binding to the orkan C quantum simulator (`../orkan`).

This module is *raw FFI plus safety rails*: it owns the library handle, mirrors
orkan's three C structs byte-exactly, guards every argument Python-side, and
converts orkan's triangular/tiled storage into dense numpy arrays.  It knows
nothing about circuits, noise models or training -- that belongs one layer up.

Why the guards matter
---------------------
orkan's ``GATE_VALIDATE`` / ``CHANNEL_VALIDATE`` macros call ``exit(EXIT_FAILURE)``
on a bad qubit index, a NULL state, or a channel on a PURE state.  There is a
``qs_error_t`` enum in ``q_types.h`` but *nothing returns it* -- it is decorative.
A missed guard here is therefore not an exception, it is **process death with no
traceback**.  The same lesson was learned twice in the Julia precedent
(``../Sturm.jl/src/orkan/ffi.jl``, bead Sturm.jl-1oy), so every wrapper below
validates before the call.

orkan's index conventions (established empirically, see `tests/test_orkan_ffi.py`)
---------------------------------------------------------------------------------
* **Qubit 0 is the least significant bit.**  ``x(state, q)`` on |0...0> moves the
  weight to storage index ``2**q``.  Native basis index is ``sum_q b_q 2**q``.
  This is the ``insertBit0(val, pos)`` convention of ``src/internal/index.h``:
  the qubit index *is* the bit position.
* **`single_from_mat` takes a COLUMN-MAJOR 2x2.**  ``mat[0]=U00, mat[1]=U10,
  mat[2]=U01, mat[3]=U11`` (read off ``src/gate/tiled/gate_tiled_1q.c``, where the
  coefficient of rho_01 is ``mat[0]*conj(mat[2])``, i.e. ``U00 conj(U01)``).
* **`two_from_mat` takes a COLUMN-MAJOR 4x4** whose basis index is
  ``2*b_hi + b_lo`` with ``hi = max(q1, q2)``, ``lo = min(q1, q2)``.  The kernel
  computes ``lo``/``hi`` itself, so **the argument order (q1, q2) is ignored**:
  ``two_from_mat(s, 0, 1, M) == two_from_mat(s, 1, 0, M)``.  A caller who wants
  "q0 is the upper tensor factor" must conjugate ``M`` by the 4x4 SWAP when
  ``q0 < q1``; :meth:`OrkanState.apply_2q_unitary` does exactly that.
* **Kraus operators are ROW-MAJOR**, concatenated ``n_terms`` at a time
  (``docs/API.md``, ``src/channel/channel.c``).  The superoperator acts on the
  row-major vectorisation: ``rho'_ab = sum_cd S[(a,b),(c,d)] rho_cd``.

Storage layouts (``docs/API.md`` "Memory Layout", ``src/state/state_tiled.c``)
-------------------------------------------------------------------------------
* ``PURE``: flat ``2**n`` amplitude vector.
* ``MIXED_PACKED``: column-major packed *lower* triangle,
  ``idx(r, c) = c*(2*dim - c + 1)/2 + (r - c)`` for ``r >= c``.
* ``MIXED_TILED``: ``TILE_DIM x TILE_DIM`` tiles, lower-triangular *at tile level*,
  row-major *within* a tile.  ``idx(r, c) = (tr*(tr+1)/2 + tc)*TILE_SIZE
  + lr*TILE_DIM + lc`` with ``tr = r >> LOG_TILE_DIM`` etc.  ``TILE_DIM`` is a
  compile-time macro (32 in the release preset, 8 in debug) and is *not* exported,
  so :func:`load_orkan` recovers it at load time from ``state_len`` of a 1-qubit
  tiled state (which is exactly ``TILE_DIM**2``).

Gotchas
-------
* ``state_init(s, n, NULL)`` **zero**-allocates.  A fresh state is the all-zero
  vector/matrix, *not* |0...0>.  :meth:`OrkanState.zero` sets element (0, 0) to 1.
* ``state_plus`` on a MIXED state gives ``|+><+|`` (every element ``2**-n``), i.e.
  a *pure* state -- despite the "maximally mixed" wording in ``include/state.h``.
* ``single_from_mat`` / ``two_from_mat`` live in ``src/gate/tiled/`` and have no
  pure or packed dispatch: they are **MIXED_TILED only**.
* ``kraus_to_superop`` returns a struct whose ``data`` was ``calloc``'d by orkan.
  There is no ``superop_free``; it must be released with ``libc``'s ``free``.
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import math
import os
from collections.abc import Iterator, Sequence
from enum import IntEnum
from pathlib import Path
from types import TracebackType

import numpy as np
import numpy.typing as npt

__all__ = [
    "OrkanState",
    "OrkanTypeError",
    "OrkanUnavailable",
    "StateType",
    "load_orkan",
    "msb0_permutation",
    "orkan_available",
]

Complex128Array = npt.NDArray[np.complex128]

#: Environment variable overriding the shared-library search.
LIB_ENV_VAR = "PTQNN_ORKAN_LIB"

#: Sibling-checkout fallback: ``<...>/Projects/orkan/cmake-build-release/src/liborkan.so``.
_SIBLING_LIB = (
    Path(__file__).resolve().parents[4] / "orkan" / "cmake-build-release" / "src" / "liborkan.so"
)

_BUILD_HINT = (
    "Build it with:\n"
    "    cd ../orkan && cmake --preset release && cmake --build --preset release\n"
    f"or point {LIB_ENV_VAR} at an existing liborkan.so."
)


class OrkanUnavailable(ImportError):  # noqa: N818 - reads as "unavailable", not an error type
    """Raised when ``liborkan.so`` cannot be located or loaded."""


class OrkanTypeError(TypeError):
    """Raised when an operation is not supported by this state's storage type.

    orkan would ``exit()`` on most of these; we raise instead.
    """


class StateType(IntEnum):
    """Mirror of ``state_type_t`` (``include/state.h``)."""

    PURE = 0
    MIXED_PACKED = 1
    MIXED_TILED = 2


# ---------------------------------------------------------------------------
# C structs -- byte-exact mirrors
# ---------------------------------------------------------------------------


class Cplx(ctypes.Structure):
    """``double _Complex``.

    x86-64 SysV classifies ``struct {double; double;}`` exactly like
    ``double _Complex`` (SSE, SSE), so libffi passes and returns it correctly.
    """

    _fields_ = (("re", ctypes.c_double), ("im", ctypes.c_double))

    @classmethod
    def of(cls, z: complex) -> Cplx:
        w = complex(z)
        return cls(w.real, w.imag)

    def value(self) -> complex:
        return complex(self.re, self.im)


class CState(ctypes.Structure):
    """``state_t`` -- 24 bytes: enum(4) + pad(4) + ptr(8) + uchar(1) + pad(7)."""

    _fields_ = (
        ("type", ctypes.c_int),
        ("data", ctypes.POINTER(Cplx)),
        ("qubits", ctypes.c_ubyte),
    )


class CKraus(ctypes.Structure):
    """``kraus_t`` -- 24 bytes: uchar(1) + pad(7) + uint64(8) + ptr(8).

    ``docs/API.md`` types ``n_terms`` as ``unsigned``; ``include/channel.h`` says
    ``uint64_t``.  The header wins (and ``sizeof`` confirms it).
    """

    _fields_ = (
        ("n_qubits", ctypes.c_ubyte),
        ("n_terms", ctypes.c_uint64),
        ("data", ctypes.POINTER(Cplx)),
    )


class CSuperop(ctypes.Structure):
    """``superop_t`` -- 16 bytes: uchar(1) + pad(7) + ptr(8)."""

    _fields_ = (("n_qubits", ctypes.c_ubyte), ("data", ctypes.POINTER(Cplx)))


_STATE_P = ctypes.POINTER(CState)
_CPLX_P = ctypes.POINTER(Cplx)

_GATES_0P = ("x", "y", "z", "h", "hy", "s", "sdg", "t", "tdg")
_GATES_1P = ("rx", "ry", "rz", "p")
_GATES_CTRL = ("cx", "cy", "cz")


# ---------------------------------------------------------------------------
# Library handle
# ---------------------------------------------------------------------------


def _candidate_paths() -> list[str]:
    env = os.environ.get(LIB_ENV_VAR)
    if env:
        # An explicit override that does not exist is an error, not a fallback.
        return [env]
    found = ctypes.util.find_library("orkan")
    return [str(_SIBLING_LIB)] + ([found] if found else [])


class _Orkan:
    """Loaded ``liborkan`` with all prototypes declared.

    Construct via :func:`load_orkan`, which caches a single instance.
    """

    def __init__(self, path: str) -> None:
        try:
            self.dll = ctypes.CDLL(path)
        except OSError as exc:  # pragma: no cover - depends on the box
            raise OrkanUnavailable(f"failed to dlopen {path!r}: {exc}\n{_BUILD_HINT}") from exc
        self.path = path
        self._libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
        self._libc.free.argtypes = [ctypes.c_void_p]
        self._libc.free.restype = None
        self._declare()
        self._assert_abi()
        self.tile_dim = self._probe_tile_dim()

    def _declare(self) -> None:
        d = self.dll
        d.state_init.argtypes = [_STATE_P, ctypes.c_ubyte, ctypes.c_void_p]
        d.state_init.restype = None
        d.state_free.argtypes = [_STATE_P]
        d.state_free.restype = None
        d.state_plus.argtypes = [_STATE_P, ctypes.c_ubyte]
        d.state_plus.restype = None
        d.state_len.argtypes = [_STATE_P]
        d.state_len.restype = ctypes.c_uint64
        d.state_get.argtypes = [_STATE_P, ctypes.c_uint64, ctypes.c_uint64]
        d.state_get.restype = Cplx
        d.state_set.argtypes = [_STATE_P, ctypes.c_uint64, ctypes.c_uint64, Cplx]
        d.state_set.restype = None

        for name in _GATES_0P:
            fn = getattr(d, name)
            fn.argtypes = [_STATE_P, ctypes.c_ubyte]
            fn.restype = None
        for name in _GATES_1P:
            fn = getattr(d, name)
            fn.argtypes = [_STATE_P, ctypes.c_ubyte, ctypes.c_double]
            fn.restype = None
        for name in _GATES_CTRL:
            fn = getattr(d, name)
            fn.argtypes = [_STATE_P, ctypes.c_ubyte, ctypes.c_ubyte]
            fn.restype = None
        # NB the C symbol is `swap_gate`, not `swap`.
        d.swap_gate.argtypes = [_STATE_P, ctypes.c_ubyte, ctypes.c_ubyte]
        d.swap_gate.restype = None
        d.ccx.argtypes = [_STATE_P, ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_ubyte]
        d.ccx.restype = None

        d.single_from_mat.argtypes = [_STATE_P, ctypes.c_ubyte, _CPLX_P]
        d.single_from_mat.restype = None
        d.two_from_mat.argtypes = [_STATE_P, ctypes.c_ubyte, ctypes.c_ubyte, _CPLX_P]
        d.two_from_mat.restype = None

        d.kraus_to_superop.argtypes = [ctypes.POINTER(CKraus)]
        d.kraus_to_superop.restype = CSuperop
        d.channel_1q.argtypes = [_STATE_P, ctypes.POINTER(CSuperop), ctypes.c_ubyte]
        d.channel_1q.restype = None

    @staticmethod
    def _assert_abi() -> None:
        sizes = {"state_t": (CState, 24), "kraus_t": (CKraus, 24), "superop_t": (CSuperop, 16)}
        for name, (cls, want) in sizes.items():
            got = ctypes.sizeof(cls)
            if got != want:
                raise OrkanUnavailable(f"ABI drift: sizeof({name}) == {got}, expected {want}")

    def _probe_tile_dim(self) -> int:
        """Recover the compile-time ``TILE_DIM`` (not an exported symbol).

        ``state_tiled_len(1) == ceil(2/TILE_DIM)*(...)/2 * TILE_SIZE == TILE_DIM**2``
        for any ``TILE_DIM >= 2``, so one allocation-free ``state_len`` call on a
        stack ``state_t`` gives it exactly.
        """
        st = CState(int(StateType.MIXED_TILED), None, 1)
        size = int(self.dll.state_len(ctypes.byref(st)))
        dim = math.isqrt(size)
        if dim * dim != size or dim & (dim - 1):
            raise OrkanUnavailable(f"cannot infer TILE_DIM from state_len == {size}")
        return dim

    def free(self, ptr: ctypes.c_void_p) -> None:
        """``libc free`` -- for ``kraus_to_superop``'s calloc'd buffer."""
        self._libc.free(ptr)


_LIB: _Orkan | None = None


def load_orkan() -> _Orkan:
    """Return the process-wide loaded ``liborkan``, loading it on first use.

    Search order: ``$PTQNN_ORKAN_LIB`` (if set, it must exist -- no silent
    fallback), then the sibling release build, then the system loader.
    """
    global _LIB
    if _LIB is None:
        candidates = _candidate_paths()
        for path in candidates:
            if os.path.exists(path):
                _LIB = _Orkan(path)
                return _LIB
        tried = ", ".join(repr(c) for c in candidates)
        raise OrkanUnavailable(f"liborkan.so not found; tried {tried}\n{_BUILD_HINT}")
    return _LIB


def orkan_available() -> bool:
    """True if :func:`load_orkan` would succeed.  Safe to call at import time."""
    try:
        load_orkan()
    except OrkanUnavailable:
        return False
    return True


# ---------------------------------------------------------------------------
# Index-convention helpers
# ---------------------------------------------------------------------------


def msb0_permutation(n: int) -> npt.NDArray[np.int64]:
    """Native (lsb0) indices in msb0 order: ``v_msb0 = v_native[msb0_permutation(n)]``.

    orkan is lsb0 (``index = sum_q b_q 2**q``); the project convention is msb0
    (``index = sum_q b_q 2**(n-1-q)``, qubit 0 = most significant).  The two differ
    by reversing the ``n``-bit string, which is an involution -- the same
    permutation converts back.
    """
    idx = np.arange(1 << n, dtype=np.int64)
    out = np.zeros_like(idx)
    for q in range(n):
        # bit at position (n-1-q) of the msb0 index -> bit position q natively
        out |= ((idx >> (n - 1 - q)) & 1) << q
    return out


_SWAP4 = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
    dtype=np.complex128,
)


def _as_cplx_buffer(a: Complex128Array) -> ctypes.Array[Cplx]:
    """Copy a complex array into a fresh C buffer of ``Cplx`` (never aliases numpy)."""
    flat = np.ascontiguousarray(a, dtype=np.complex128).ravel()
    buf = (Cplx * flat.size)()
    view = np.frombuffer(buf, dtype=np.complex128, count=flat.size)
    view[:] = flat
    return buf


def _row_chunk(dim: int) -> int:
    """Rows per pass in the dense reconstruction, capping the index array at ~64 MB."""
    return max(1, min(dim, (1 << 23) // max(1, dim)))


# ---------------------------------------------------------------------------
# The state wrapper
# ---------------------------------------------------------------------------


class OrkanState:
    """An owned orkan ``state_t`` with guarded gates and dense readout.

    Use as a context manager (or call :meth:`free`); ``__del__`` is only a
    backstop.  Every method raises rather than letting orkan ``exit()``.
    """

    __slots__ = ("_freed", "_lib", "_st")

    def __init__(self, n_qubits: int, kind: StateType = StateType.PURE) -> None:
        """Allocate an **all-zero** state (orkan's raw ``state_init`` semantics).

        Prefer :meth:`zero` or :meth:`plus`; a raw all-zero state is not a valid
        quantum state.
        """
        if not 1 <= int(n_qubits) <= 255:
            raise ValueError(f"n_qubits must be in [1, 255], got {n_qubits}")
        self._lib = load_orkan()
        self._freed = False
        self._st = CState(int(StateType(kind)), None, 0)
        self._lib.dll.state_init(ctypes.byref(self._st), ctypes.c_ubyte(int(n_qubits)), None)
        if not self._st.data:
            self._freed = True
            raise MemoryError(f"orkan state_init failed for {n_qubits} qubits ({kind.name})")

    # -- constructors -------------------------------------------------------

    @classmethod
    def zero(cls, n_qubits: int, kind: StateType = StateType.PURE) -> OrkanState:
        """|0...0> (PURE) or |0...0><0...0| (MIXED).

        ``state_init`` zero-allocates, so the |0...0> amplitude/population has to
        be written explicitly -- exactly as ``Sturm.jl``'s ``_state_new`` does.
        """
        st = cls(n_qubits, kind)
        st._lib.dll.state_set(ctypes.byref(st._st), 0, 0, Cplx.of(1.0))
        return st

    @classmethod
    def plus(cls, n_qubits: int, kind: StateType = StateType.PURE) -> OrkanState:
        """|+>^n (PURE) or ``|+><+|`` (MIXED -- pure, despite the header's wording)."""
        if not 1 <= int(n_qubits) <= 255:
            raise ValueError(f"n_qubits must be in [1, 255], got {n_qubits}")
        st = cls.__new__(cls)
        st._lib = load_orkan()
        st._freed = False
        st._st = CState(int(StateType(kind)), None, 0)
        st._lib.dll.state_plus(ctypes.byref(st._st), ctypes.c_ubyte(int(n_qubits)))
        if not st._st.data:
            st._freed = True
            raise MemoryError(f"orkan state_plus failed for {n_qubits} qubits ({kind.name})")
        return st

    # -- lifetime -----------------------------------------------------------

    @property
    def n_qubits(self) -> int:
        return int(self._st.qubits)

    @property
    def kind(self) -> StateType:
        return StateType(self._st.type)

    @property
    def freed(self) -> bool:
        return self._freed

    def free(self) -> None:
        """Release the C buffer.  Idempotent."""
        if not self._freed:
            self._lib.dll.state_free(ctypes.byref(self._st))
            self._freed = True

    def __enter__(self) -> OrkanState:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.free()

    def __del__(self) -> None:  # pragma: no cover - GC timing
        # Backstop only: interpreter shutdown may already have torn down ctypes.
        with contextlib.suppress(Exception):
            self.free()

    def __repr__(self) -> str:
        state = "freed" if self._freed else f"{self.n_qubits}q"
        return f"<OrkanState {self.kind.name} {state}>"

    # -- guards -------------------------------------------------------------

    def _live(self) -> None:
        if self._freed:
            raise RuntimeError("OrkanState has been freed")

    def _q(self, *qs: int) -> tuple[int, ...]:
        """Validate qubit indices; orkan would ``exit()`` on a bad one."""
        self._live()
        n = self.n_qubits
        out = []
        for q in qs:
            qi = int(q)
            if not 0 <= qi < n:
                raise IndexError(f"qubit {q} out of range [0, {n})")
            out.append(qi)
        if len(set(out)) != len(out):
            raise ValueError(f"qubits must be distinct, got {qs}")
        return tuple(out)

    def _require_tiled(self, what: str) -> None:
        if self.kind is not StateType.MIXED_TILED:
            raise OrkanTypeError(
                f"{what} is implemented only in orkan's tiled backend "
                f"(src/gate/tiled/); state is {self.kind.name}"
            )

    def _require_mixed(self, what: str) -> None:
        if self.kind is StateType.PURE:
            raise OrkanTypeError(f"{what} requires a mixed state; orkan exit()s on PURE")

    # -- native gates -------------------------------------------------------
    # All of these dispatch to pure/packed/tiled kernels: every state type works.

    def _g0(self, name: str, q: int) -> None:
        (qi,) = self._q(q)
        getattr(self._lib.dll, name)(ctypes.byref(self._st), ctypes.c_ubyte(qi))

    def _g1(self, name: str, q: int, theta: float) -> None:
        (qi,) = self._q(q)
        getattr(self._lib.dll, name)(
            ctypes.byref(self._st), ctypes.c_ubyte(qi), ctypes.c_double(float(theta))
        )

    def x(self, q: int) -> None:
        """Pauli X."""
        self._g0("x", q)

    def y(self, q: int) -> None:
        """Pauli Y."""
        self._g0("y", q)

    def z(self, q: int) -> None:
        """Pauli Z."""
        self._g0("z", q)

    def h(self, q: int) -> None:
        """Hadamard."""
        self._g0("h", q)

    def hy(self, q: int) -> None:
        """``Hy = (1/sqrt 2)[[1, -1], [1, 1]]`` (orkan-specific, not a standard name)."""
        self._g0("hy", q)

    def s(self, q: int) -> None:
        """``S = diag(1, i)``."""
        self._g0("s", q)

    def sdg(self, q: int) -> None:
        """``S^dagger = diag(1, -i)``."""
        self._g0("sdg", q)

    def t(self, q: int) -> None:
        """``T = diag(1, e^{i pi/4})``."""
        self._g0("t", q)

    def tdg(self, q: int) -> None:
        """``T^dagger``."""
        self._g0("tdg", q)

    def rx(self, q: int, theta: float) -> None:
        """``exp(-i theta X / 2)``."""
        self._g1("rx", q, theta)

    def ry(self, q: int, theta: float) -> None:
        """``exp(-i theta Y / 2)``."""
        self._g1("ry", q, theta)

    def rz(self, q: int, theta: float) -> None:
        """``exp(-i theta Z / 2) = diag(e^{-i theta/2}, e^{i theta/2})``."""
        self._g1("rz", q, theta)

    def p(self, q: int, theta: float) -> None:
        """``P(theta) = diag(1, e^{i theta})``."""
        self._g1("p", q, theta)

    def cx(self, control: int, target: int) -> None:
        """CNOT."""
        c, t = self._q(control, target)
        self._lib.dll.cx(ctypes.byref(self._st), ctypes.c_ubyte(c), ctypes.c_ubyte(t))

    def cy(self, control: int, target: int) -> None:
        """Controlled-Y."""
        c, t = self._q(control, target)
        self._lib.dll.cy(ctypes.byref(self._st), ctypes.c_ubyte(c), ctypes.c_ubyte(t))

    def cz(self, control: int, target: int) -> None:
        """Controlled-Z."""
        c, t = self._q(control, target)
        self._lib.dll.cz(ctypes.byref(self._st), ctypes.c_ubyte(c), ctypes.c_ubyte(t))

    def swap(self, q1: int, q2: int) -> None:
        """SWAP (C symbol ``swap_gate``)."""
        a, b = self._q(q1, q2)
        self._lib.dll.swap_gate(ctypes.byref(self._st), ctypes.c_ubyte(a), ctypes.c_ubyte(b))

    def ccx(self, control1: int, control2: int, target: int) -> None:
        """Toffoli."""
        c1, c2, t = self._q(control1, control2, target)
        self._lib.dll.ccx(
            ctypes.byref(self._st),
            ctypes.c_ubyte(c1),
            ctypes.c_ubyte(c2),
            ctypes.c_ubyte(t),
        )

    # -- arbitrary unitaries (MIXED_TILED only) -----------------------------

    def apply_1q_unitary(self, q: int, u: Complex128Array) -> None:
        """``rho -> U rho U^dagger`` for a 2x2 ``u`` on qubit ``q``.

        Wraps ``single_from_mat``, which exists **only** in the tiled backend
        (``src/gate/tiled/gate_tiled_1q.c``); PURE and MIXED_PACKED raise
        :class:`OrkanTypeError`.  ``u`` is an ordinary row-major numpy 2x2 -- the
        column-major transpose orkan wants is done here.
        """
        (qi,) = self._q(q)
        self._require_tiled("apply_1q_unitary (single_from_mat)")
        mat = np.asarray(u, dtype=np.complex128)
        if mat.shape != (2, 2):
            raise ValueError(f"expected a 2x2 matrix, got shape {mat.shape}")
        buf = _as_cplx_buffer(mat.T)  # row-major transpose == column-major
        self._lib.dll.single_from_mat(ctypes.byref(self._st), ctypes.c_ubyte(qi), buf)

    def apply_2q_unitary(self, q0: int, q1: int, u: Complex128Array) -> None:
        """``rho -> U rho U^dagger`` for a 4x4 ``u`` on qubits ``(q0, q1)``.

        ``u`` is indexed ``2*b_q0 + b_q1``: **q0 is the upper tensor factor**, so
        ``u = kron(A, B)`` means ``A`` on ``q0`` and ``B`` on ``q1``, and
        ``CRX = |0><0| (x) I + |1><1| (x) RX`` is a CRX with control ``q0``.

        orkan's ``two_from_mat`` instead hard-codes ``2*b_max + b_min`` and ignores
        the order of its two qubit arguments, so when ``q0 < q1`` we pass
        ``SWAP4 @ u @ SWAP4`` -- the same operator re-expressed in orkan's basis.

        Tiled-only, like :meth:`apply_1q_unitary`.
        """
        a, b = self._q(q0, q1)
        self._require_tiled("apply_2q_unitary (two_from_mat)")
        mat = np.asarray(u, dtype=np.complex128)
        if mat.shape != (4, 4):
            raise ValueError(f"expected a 4x4 matrix, got shape {mat.shape}")
        if a < b:
            mat = _SWAP4 @ mat @ _SWAP4
        buf = _as_cplx_buffer(mat.T)
        self._lib.dll.two_from_mat(
            ctypes.byref(self._st), ctypes.c_ubyte(a), ctypes.c_ubyte(b), buf
        )

    # -- channels (MIXED only) ---------------------------------------------

    def apply_kraus_1q(self, q: int, kraus: Sequence[Complex128Array]) -> None:
        """``rho -> sum_k K_k rho K_k^dagger`` for 2x2 Kraus operators on qubit ``q``.

        Goes through orkan's ``kraus_to_superop`` (row-major concatenated Kraus
        matrices) and ``channel_1q``.  Trace preservation is *not* checked -- if
        ``sum_k K^dagger K != I`` orkan will happily apply a non-TP map.

        MIXED only: ``channel_1q`` ``exit()``s on PURE states.  Both mixed backends
        (packed and tiled) are supported.
        """
        (qi,) = self._q(q)
        self._require_mixed("apply_kraus_1q (channel_1q)")
        ops = [np.asarray(k, dtype=np.complex128) for k in kraus]
        if not ops:
            raise ValueError("need at least one Kraus operator")
        for k in ops:
            if k.shape != (2, 2):
                raise ValueError(f"1-qubit Kraus operators must be 2x2, got {k.shape}")
        flat = np.concatenate([k.reshape(-1) for k in ops])  # row-major, concatenated
        kbuf = _as_cplx_buffer(flat)
        ck = CKraus(1, len(ops), ctypes.cast(kbuf, _CPLX_P))
        sop = self._lib.dll.kraus_to_superop(ctypes.byref(ck))
        if not sop.data:
            raise MemoryError("orkan kraus_to_superop failed")
        try:
            self._lib.dll.channel_1q(ctypes.byref(self._st), ctypes.byref(sop), ctypes.c_ubyte(qi))
        finally:
            # No superop_free in orkan; the buffer is a plain calloc.
            self._lib.free(ctypes.cast(sop.data, ctypes.c_void_p))

    # -- readout ------------------------------------------------------------

    def get(self, row: int, col: int = 0) -> complex:
        """One element via ``state_get`` (Hermitian-aware for MIXED; ``col`` unused for PURE)."""
        self._live()
        dim = 1 << self.n_qubits
        if not (0 <= row < dim and 0 <= col < dim):
            raise IndexError(f"({row}, {col}) out of range for dim {dim}")
        if self.kind is StateType.PURE:
            col = 0
        z: Cplx = self._lib.dll.state_get(ctypes.byref(self._st), row, col)
        return z.value()

    def _raw(self) -> Complex128Array:
        """Zero-copy numpy view of orkan's storage array (valid until :meth:`free`)."""
        n_elem = int(self._lib.dll.state_len(ctypes.byref(self._st)))
        buf = ctypes.cast(self._st.data, ctypes.POINTER(ctypes.c_double * (2 * n_elem)))
        flat = np.frombuffer(buf.contents, dtype=np.float64)
        arr: Complex128Array = flat.view(np.complex128)
        return arr

    def to_numpy(self, order: str = "lsb0") -> Complex128Array:
        """Dense readout: the statevector (PURE) or the full density matrix (MIXED).

        The MIXED triangular/tiled buffer is expanded to the whole Hermitian matrix
        here (stored element for ``row >= col``, its conjugate above the diagonal),
        so the result satisfies ``rho == rho.conj().T`` exactly up to the stored
        values.

        ``order``:
          * ``"lsb0"`` (default) -- orkan's native basis, ``index = sum_q b_q 2**q``
            (qubit 0 = least significant bit).
          * ``"msb0"`` -- the project convention, ``index = sum_q b_q 2**(n-1-q)``
            (qubit 0 = most significant bit).  Applied as a bit-reversal
            permutation of the basis (both axes, for a density matrix).
        """
        self._live()
        if order not in ("lsb0", "msb0"):
            raise ValueError(f"order must be 'lsb0' or 'msb0', got {order!r}")
        n = self.n_qubits
        dim = 1 << n
        raw = self._raw()

        out: Complex128Array
        if self.kind is StateType.PURE:
            out = np.array(raw[:dim], dtype=np.complex128)
            if order == "msb0":
                out = out[msb0_permutation(n)]
            return out

        out = np.empty((dim, dim), dtype=np.complex128)
        cols = np.arange(dim, dtype=np.int64)
        for start, rows in self._row_blocks(dim):
            hi = np.maximum(rows, cols)
            lo = np.minimum(rows, cols)
            idx = self._storage_index(dim, hi, lo)
            vals = raw[idx]
            out[start : start + rows.shape[0]] = np.where(rows >= cols, vals, vals.conj())
        if order == "msb0":
            perm = msb0_permutation(n)
            out = out[np.ix_(perm, perm)]
        return out

    @staticmethod
    def _row_blocks(dim: int) -> Iterator[tuple[int, npt.NDArray[np.int64]]]:
        step = _row_chunk(dim)
        for start in range(0, dim, step):
            stop = min(start + step, dim)
            yield start, np.arange(start, stop, dtype=np.int64)[:, None]

    def _storage_index(
        self, dim: int, hi: npt.NDArray[np.int64], lo: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.int64]:
        """Flat buffer index of the stored lower-triangle element ``(hi, lo)``."""
        packed: npt.NDArray[np.int64]
        if self.kind is StateType.MIXED_PACKED:
            # column-major packed: lo*(2*dim - lo + 1)/2 + (hi - lo)
            packed = lo * (2 * dim - lo + 1) // 2 + (hi - lo)
            return packed
        tile = self._lib.tile_dim
        log_tile = tile.bit_length() - 1
        tr = hi >> log_tile
        tc = lo >> log_tile
        packed = (
            (tr * (tr + 1) // 2 + tc) * (tile * tile) + (hi & (tile - 1)) * tile + (lo & (tile - 1))
        )
        return packed
