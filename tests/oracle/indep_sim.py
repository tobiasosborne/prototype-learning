"""Small, independent full-matrix density-matrix simulator for cross-check tests.

This module deliberately uses explicit Kronecker products and permutation matrices.  It does
not import the production simulator, so agreement with it is meaningful.  Qubit zero is the
most-significant tensor factor throughout, as required by the oracle task specification.
"""

from collections.abc import Sequence
from numbers import Real

import numpy as np
import numpy.typing as npt

type ComplexArray = npt.NDArray[np.complex128]
type RealArray = npt.NDArray[np.float64]
type Operation = tuple[object, ...]

__all__ = ["embed", "marginal", "probabilities", "run"]


_I = np.eye(2, dtype=np.complex128)
_H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
_S = np.diag(np.array([1.0, 1.0j], dtype=np.complex128))
_T = np.diag(np.array([1.0, np.exp(1.0j * np.pi / 4.0)], dtype=np.complex128))
_P0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
_P1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
_RESET_0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
_RESET_1 = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
_SWAP = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.complex128,
)

_NAMED_GATES: dict[str, ComplexArray] = {
    "h": _H,
    "x": _X,
    "y": _Y,
    "z": _Z,
    "s": _S,
    "t": _T,
}


def _validate_n(n: int) -> None:
    if not isinstance(n, int):
        raise TypeError("n must be an int")
    if n < 0:
        raise ValueError("n must be non-negative")


def _validate_qubits(n: int, qubits: Sequence[int]) -> None:
    if len(set(qubits)) != len(qubits):
        raise ValueError("qubits must be distinct")
    if any(q < 0 or q >= n for q in qubits):
        raise ValueError(f"qubits must be in range(0, {n})")


def _as_qubit(value: object) -> int:
    if not isinstance(value, int):
        raise TypeError("a qubit index must be an int")
    return value


def _as_qubit_list(value: object) -> list[int]:
    if not isinstance(value, list):
        raise TypeError("qubit collections in operations must be lists")
    return [_as_qubit(q) for q in value]


def _as_angle(value: object) -> float:
    if not isinstance(value, Real):
        raise TypeError("rotation angles must be real numbers")
    return float(value)


def _as_matrix(value: object) -> ComplexArray:
    matrix = np.asarray(value, dtype=np.complex128)
    if matrix.ndim != 2:
        raise ValueError("an operator must be a matrix")
    return matrix


def _expect_length(op: Operation, expected: int) -> None:
    if len(op) != expected:
        raise ValueError(f"operation {op[0]!r} must have {expected} entries")


def _index_to_bits(index: int, width: int) -> tuple[int, ...]:
    if width == 0:
        return ()
    return tuple(int(bit) for bit in np.unravel_index(index, (2,) * width))


def _bits_to_index(bits: Sequence[int]) -> int:
    width = len(bits)
    return sum(bit * 2 ** (width - 1 - position) for position, bit in enumerate(bits))


def _matrix_qubit_count(matrix: ComplexArray) -> int:
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("density matrix must be square")
    dimension = matrix.shape[0]
    qubits = 0
    expected_dimension = 1
    while expected_dimension < dimension:
        expected_dimension *= 2
        qubits += 1
    if expected_dimension != dimension:
        raise ValueError("density-matrix dimension must be a power of two")
    return qubits


def embed(n: int, qubits: Sequence[int], M: object) -> ComplexArray:
    """Embed ``M`` on ``qubits`` using an explicit tensor-factor permutation.

    ``M`` uses the listed-qubit ordering, with the first listed qubit as its most-significant
    tensor factor.  The returned operator uses canonical qubit order ``0, ..., n - 1``.
    """

    _validate_n(n)
    listed_qubits = list(qubits)
    _validate_qubits(n, listed_qubits)
    matrix = _as_matrix(M)
    local_dimension = 2 ** len(listed_qubits)
    if matrix.shape != (local_dimension, local_dimension):
        raise ValueError(
            f"operator on {len(listed_qubits)} qubits must have shape "
            f"({local_dimension}, {local_dimension})"
        )

    remaining_qubits = [q for q in range(n) if q not in listed_qubits]
    ordered_qubits = listed_qubits + remaining_qubits
    identity_on_rest = np.eye(2 ** len(remaining_qubits), dtype=np.complex128)
    operator_in_listed_order = np.kron(matrix, identity_on_rest)

    dimension = 2**n
    permutation = np.zeros((dimension, dimension), dtype=np.complex128)
    for ordered_index in range(dimension):
        ordered_bits = _index_to_bits(ordered_index, n)
        canonical_bits = [0] * n
        for position, qubit in enumerate(ordered_qubits):
            canonical_bits[qubit] = ordered_bits[position]
        canonical_index = _bits_to_index(canonical_bits)
        permutation[canonical_index, ordered_index] = 1.0

    embedded = permutation @ operator_in_listed_order @ permutation.conj().T
    return np.asarray(embedded, dtype=np.complex128)


def _rx(theta: float) -> ComplexArray:
    # Standard rotations and qubit ordering are fixed by [TASK SPEC].
    cosine = np.cos(theta / 2.0)
    sine = np.sin(theta / 2.0)
    return np.array([[cosine, -1.0j * sine], [-1.0j * sine, cosine]], dtype=np.complex128)


def _ry(theta: float) -> ComplexArray:
    cosine = np.cos(theta / 2.0)
    sine = np.sin(theta / 2.0)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=np.complex128)


def _rz(theta: float) -> ComplexArray:
    return np.diag(
        np.array(
            [np.exp(-1.0j * theta / 2.0), np.exp(1.0j * theta / 2.0)],
            dtype=np.complex128,
        )
    )


def _controlled(n: int, control: int, target: int, target_gate: ComplexArray) -> ComplexArray:
    if control == target:
        raise ValueError("control and target must be distinct")
    _validate_qubits(n, [control, target])
    control_zero = embed(n, [control], _P0)
    control_one = embed(n, [control], _P1)
    gate_on_target = embed(n, [target], target_gate)
    return control_zero + control_one @ gate_on_target


def _apply_unitary(rho: ComplexArray, unitary: ComplexArray) -> ComplexArray:
    return unitary @ rho @ unitary.conj().T


def _apply_kraus(rho: ComplexArray, operators: Sequence[ComplexArray]) -> ComplexArray:
    result = np.zeros_like(rho, dtype=np.complex128)
    for operator in operators:
        result = result + operator @ rho @ operator.conj().T
    return result


def run(n: int, ops: list[Operation], rho0: ComplexArray | None = None) -> ComplexArray:
    """Apply ``ops`` in order and return the final full density matrix."""

    _validate_n(n)
    dimension = 2**n
    if rho0 is None:
        rho = np.zeros((dimension, dimension), dtype=np.complex128)
        rho[0, 0] = 1.0
    else:
        rho = np.asarray(rho0, dtype=np.complex128).copy()
        if rho.shape != (dimension, dimension):
            raise ValueError(f"rho0 must have shape ({dimension}, {dimension})")

    for op in ops:
        if not op or not isinstance(op[0], str):
            raise ValueError("every operation must start with a string name")
        name = op[0]

        if name in _NAMED_GATES:
            _expect_length(op, 2)
            qubit = _as_qubit(op[1])
            rho = _apply_unitary(rho, embed(n, [qubit], _NAMED_GATES[name]))
        elif name in {"rx", "ry", "rz"}:
            _expect_length(op, 3)
            qubit = _as_qubit(op[1])
            theta = _as_angle(op[2])
            rotation = {"rx": _rx, "ry": _ry, "rz": _rz}[name](theta)
            rho = _apply_unitary(rho, embed(n, [qubit], rotation))
        elif name == "crx":
            _expect_length(op, 4)
            control = _as_qubit(op[1])
            target = _as_qubit(op[2])
            theta = _as_angle(op[3])
            rho = _apply_unitary(rho, _controlled(n, control, target, _rx(theta)))
        elif name in {"cx", "cz"}:
            _expect_length(op, 3)
            control = _as_qubit(op[1])
            target = _as_qubit(op[2])
            target_gate = _X if name == "cx" else _Z
            rho = _apply_unitary(rho, _controlled(n, control, target, target_gate))
        elif name == "swap":
            _expect_length(op, 3)
            first = _as_qubit(op[1])
            second = _as_qubit(op[2])
            if first == second:
                raise ValueError("swap qubits must be distinct")
            rho = _apply_unitary(rho, embed(n, [first, second], _SWAP))
        elif name == "u":
            _expect_length(op, 3)
            qubits = _as_qubit_list(op[1])
            unitary = embed(n, qubits, _as_matrix(op[2]))
            rho = _apply_unitary(rho, unitary)
        elif name == "kraus":
            _expect_length(op, 3)
            qubits = _as_qubit_list(op[1])
            if not isinstance(op[2], list) or not op[2]:
                raise TypeError("a Kraus operation needs a non-empty list of operators")
            full_operators = [embed(n, qubits, _as_matrix(operator)) for operator in op[2]]
            rho = _apply_kraus(rho, full_operators)
        elif name == "reset":
            _expect_length(op, 2)
            qubit = _as_qubit(op[1])
            full_operators = [embed(n, [qubit], _RESET_0), embed(n, [qubit], _RESET_1)]
            rho = _apply_kraus(rho, full_operators)
        else:
            raise ValueError(f"unsupported operation {name!r}")

    return np.asarray(rho, dtype=np.complex128)


def probabilities(rho: ComplexArray) -> RealArray:
    """Return computational-basis probabilities in canonical MSB-first order."""

    matrix = np.asarray(rho, dtype=np.complex128)
    _matrix_qubit_count(matrix)
    return np.asarray(np.diag(matrix).real, dtype=np.float64)


def marginal(rho: ComplexArray, keep: list[int]) -> ComplexArray:
    """Trace out all but ``keep``, preserving the given order of kept qubits."""

    matrix = np.asarray(rho, dtype=np.complex128)
    n = _matrix_qubit_count(matrix)
    _validate_qubits(n, keep)
    traced_qubits = [q for q in range(n) if q not in keep]
    kept_dimension = 2 ** len(keep)
    traced_dimension = 2 ** len(traced_qubits)
    result = np.zeros((kept_dimension, kept_dimension), dtype=np.complex128)

    for kept_row in range(kept_dimension):
        kept_row_bits = _index_to_bits(kept_row, len(keep))
        for kept_column in range(kept_dimension):
            kept_column_bits = _index_to_bits(kept_column, len(keep))
            total = 0.0 + 0.0j
            for traced_index in range(traced_dimension):
                traced_bits = _index_to_bits(traced_index, len(traced_qubits))
                canonical_row = [0] * n
                canonical_column = [0] * n
                for position, qubit in enumerate(keep):
                    canonical_row[qubit] = kept_row_bits[position]
                    canonical_column[qubit] = kept_column_bits[position]
                for position, qubit in enumerate(traced_qubits):
                    canonical_row[qubit] = traced_bits[position]
                    canonical_column[qubit] = traced_bits[position]
                total += matrix[
                    _bits_to_index(canonical_row),
                    _bits_to_index(canonical_column),
                ]
            result[kept_row, kept_column] = total

    return result
