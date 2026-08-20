"""Circuit intermediate representation: every op is a completely positive map.

Design (borrowed from ``../Sturm.jl``: "the IR represents channels, not unitaries"): a circuit
is an ordered list of :class:`Op`\\ s acting on named qubits.  :class:`Gate` is a unitary op and
carries an explicit *unitarity witness* (checked at construction); :class:`Channel` is a general
CPTP map in Kraus form (trace preservation checked at construction).  Backends consume ops
through ``as_channel()`` when they need uniformity, or special-case gates for speed.

Conventions
-----------
* Qubit 0 is the **most significant** bit of a computational-basis index:
  ``index = sum(b_q << (n - 1 - q))``.  The paper's circuit figures number qubits 1..n top to
  bottom; our qubit ``q`` is the paper's ``q+1``.  Bit strings such as ``01101`` in
  ``[PTQNN Fig 5]`` read left to right as qubits 0..4 (first input first).
* Multi-qubit matrices are written in the tensor order of ``op.qubits`` (first listed qubit =
  most significant).  For ``crx(c, t, θ)`` the matrix is ``|0⟩⟨0|_c ⊗ I + |1⟩⟨1|_c ⊗ RX(θ)_t``.
* ``RX(θ) = exp(-iθX/2)``.  The stochastic neuron's activation law is
  ``P(1) = sin²(θ/2)`` for ``RX(θ)|0⟩``; the paper's perceptron "controlled rotation (CRX)
  gates ... lead to a probabilistic activation of the third qubit" ``[PTQNN §II.B, Fig 1]``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

Complex = npt.NDArray[np.complex128]

_ATOL = 1e-10


def _as_complex(m: npt.ArrayLike) -> Complex:
    """A private, read-only complex128 copy — ops never alias caller arrays or module constants."""
    out = np.array(m, dtype=np.complex128, order="C", copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True, eq=False)
class Gate:
    """A unitary acting on ``qubits``; ``matrix`` is ``2^k × 2^k`` in the order of ``qubits``."""

    qubits: tuple[int, ...]
    matrix: Complex
    name: str
    params: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        m = _as_complex(self.matrix)
        object.__setattr__(self, "matrix", m)
        object.__setattr__(self, "qubits", tuple(int(q) for q in self.qubits))
        k = len(self.qubits)
        if len(set(self.qubits)) != k:
            raise ValueError(f"qubits must be distinct, got {self.qubits}")
        if m.shape != (2**k, 2**k):
            raise ValueError(f"matrix shape {m.shape} does not match {k} qubits")
        if not np.allclose(m.conj().T @ m, np.eye(2**k), atol=_ATOL):
            raise ValueError(f"matrix for gate {self.name!r} is not unitary")

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Gate)
            and self.qubits == other.qubits
            and self.name == other.name
            and self.params == other.params
            and np.array_equal(self.matrix, other.matrix)
        )

    def __hash__(self) -> int:
        return hash((self.qubits, self.name, self.params))

    @property
    def n_qubits(self) -> int:
        return len(self.qubits)

    def as_channel(self) -> Channel:
        return Channel(qubits=self.qubits, kraus=[self.matrix], name=self.name)

    def dagger(self) -> Gate:
        """Inverse gate.  For the parametrised rotations the parameter is negated."""
        params = tuple(-p for p in self.params) if self.name in ("rx", "ry", "rz", "crx") else ()
        return Gate(self.qubits, self.matrix.conj().T, self.name + ("" if params else "†"), params)


@dataclass(frozen=True, eq=False)
class Channel:
    """A CPTP map ``ρ ↦ Σ_i K_i ρ K_i†`` on ``qubits``.  Each ``K_i`` is ``2^k × 2^k``."""

    qubits: tuple[int, ...]
    kraus: list[Complex]
    name: str

    def __post_init__(self) -> None:
        ks = [_as_complex(k) for k in self.kraus]
        object.__setattr__(self, "kraus", ks)
        object.__setattr__(self, "qubits", tuple(int(q) for q in self.qubits))
        k = len(self.qubits)
        if len(set(self.qubits)) != k:
            raise ValueError(f"qubits must be distinct, got {self.qubits}")
        d = 2**k
        for m in ks:
            if m.shape != (d, d):
                raise ValueError(f"Kraus shape {m.shape} does not match {k} qubits")
        s = sum(m.conj().T @ m for m in ks)
        if not np.allclose(s, np.eye(d), atol=_ATOL):
            raise ValueError(f"Kraus set for {self.name!r} is not trace preserving (Σ K†K ≠ I)")

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Channel)
            and self.qubits == other.qubits
            and self.name == other.name
            and len(self.kraus) == len(other.kraus)
            and all(np.array_equal(a, b) for a, b in zip(self.kraus, other.kraus, strict=True))
        )

    def __hash__(self) -> int:
        return hash((self.qubits, self.name, len(self.kraus)))

    @property
    def n_qubits(self) -> int:
        return len(self.qubits)

    def as_channel(self) -> Channel:
        return self


Op = Gate | Channel


# --- gate factories -------------------------------------------------------------------------


def _trusted(
    qubits: tuple[int, ...], matrix: Complex, name: str, params: tuple[float, ...]
) -> Gate:
    """Construct a ``Gate`` whose unitarity is a theorem (rotations, Paulis, ``controlled`` of
    those) without the ``O(8^k)`` witness check — it was 55 % of a training loss evaluation
    (``reviews/001`` #6).  ``unitary()`` and direct ``Gate(...)`` calls still verify."""
    if len(set(qubits)) != len(qubits):
        raise ValueError(f"qubits must be distinct, got {qubits}")
    g = Gate.__new__(Gate)
    object.__setattr__(g, "qubits", qubits)
    object.__setattr__(g, "matrix", _as_complex(matrix))
    object.__setattr__(g, "name", name)
    object.__setattr__(g, "params", params)
    return g


def _rot(axis: str, theta: float) -> Complex:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    if axis == "x":
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
    if axis == "y":
        return np.array([[c, -s], [s, c]], dtype=np.complex128)
    return np.array([[np.exp(-0.5j * theta), 0], [0, np.exp(0.5j * theta)]], dtype=np.complex128)


def rx(q: int, theta: float) -> Gate:
    """``RX(θ) = exp(-iθX/2)``; ``P(1) = sin²(θ/2)`` on ``|0⟩`` ``[SNN §III]``."""
    return _trusted((q,), _rot("x", theta), "rx", (float(theta),))


def ry(q: int, theta: float) -> Gate:
    return _trusted((q,), _rot("y", theta), "ry", (float(theta),))


def rz(q: int, theta: float) -> Gate:
    return _trusted((q,), _rot("z", theta), "rz", (float(theta),))


_H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def h(q: int) -> Gate:
    """Hadamard — "used to encode all possible input states in superposition" ``[PTQNN Fig 5]``."""
    return _trusted((q,), _H, "h", ())


def x(q: int) -> Gate:
    return _trusted((q,), _X, "x", ())


def y(q: int) -> Gate:
    return _trusted((q,), _Y, "y", ())


def z(q: int) -> Gate:
    return _trusted((q,), _Z, "z", ())


def controlled(u: Complex) -> Complex:
    """``|0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ U`` with the control as the most significant factor."""
    d = u.shape[0]
    m = np.eye(2 * d, dtype=np.complex128)
    m[d:, d:] = u
    return m


def crx(control: int, target: int, theta: float) -> Gate:
    """Controlled ``RX(θ)``: the perceptron's synapse ``[PTQNN Fig 1, Fig 2]``."""
    if control == target:
        raise ValueError("control and target must be distinct")
    return _trusted((control, target), controlled(_rot("x", theta)), "crx", (float(theta),))


def cx(control: int, target: int) -> Gate:
    if control == target:
        raise ValueError("control and target must be distinct")
    return _trusted((control, target), controlled(_X), "cx", ())


def unitary(qubits: Sequence[int], matrix: npt.ArrayLike, name: str = "u") -> Gate:
    return Gate(tuple(qubits), _as_complex(matrix), name)


# --- circuit ---------------------------------------------------------------------------------


@dataclass
class Circuit:
    """An ordered list of ops on ``n_qubits`` qubits.  Mutable builder; backends read ``ops``."""

    n_qubits: int
    ops: list[Op] = field(default_factory=list)
    initial_state: Complex | None = None
    """Optional normalised statevector the circuit starts from instead of ``|0…0⟩`` — used to
    load a whole dataset as a superposition ("quantum state preparation can be used, as suggested
    in [8, 20, 21]" ``[PTQNN §III.C]``).  Backends that cannot prepare arbitrary states must
    synthesise it; simulators just start from it."""

    def __post_init__(self) -> None:
        ops, self.ops = list(self.ops), []
        for op in ops:
            self.append(op)
        if self.initial_state is not None:
            v = _as_complex(self.initial_state).ravel()
            if v.shape != (2**self.n_qubits,):
                raise ValueError("initial_state has the wrong dimension")
            if not np.isclose(np.vdot(v, v).real, 1.0, atol=_ATOL):
                raise ValueError("initial_state is not normalised")
            self.initial_state = v

    def append(self, op: Op) -> Circuit:
        for q in op.qubits:
            if not 0 <= q < self.n_qubits:
                raise ValueError(f"qubit {q} out of range for {self.n_qubits}-qubit circuit")
        self.ops.append(op)
        return self

    def extend(self, ops: Iterable[Op]) -> Circuit:
        for op in ops:
            self.append(op)
        return self

    def count(self, name: str) -> int:
        return sum(1 for op in self.ops if op.name == name)

    def n_two_qubit_gates(self) -> int:
        return sum(1 for op in self.ops if isinstance(op, Gate) and op.n_qubits == 2)

    def gates(self) -> list[Gate]:
        return [op for op in self.ops if isinstance(op, Gate)]

    def is_unitary(self) -> bool:
        return all(isinstance(op, Gate) for op in self.ops)
