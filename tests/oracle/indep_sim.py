"""Independent, deliberately direct density-matrix simulator for test cross-checks.

Qubit 0 is the most significant tensor factor.  Operators are embedded by explicitly
constructing ``M kron I_rest`` and conjugating it with a basis permutation matrix; this
module intentionally shares no simulator machinery with :mod:`ptqnn`.
"""

from numbers import Real
from typing import cast

import numpy as np
import numpy.typing as npt

ComplexMatrix = npt.NDArray[np.complex128]
RealVector = npt.NDArray[np.float64]
Operation = tuple[object, ...]

_I = np.eye(2, dtype=np.complex128)
_P0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
_P1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)


def _basis_bits(index: int, width: int) -> tuple[int, ...]:
    """Return the msb-first bits of a basis index without bit-level manipulation."""
    if width == 0:
        return ()
    return tuple(int(bit) for bit in np.unravel_index(index, (2,) * width))


def _basis_index(bits: tuple[int, ...]) -> int:
    """Return the basis index of an msb-first bit tuple."""
    if not bits:
        return 0
    return int(np.ravel_multi_index(bits, (2,) * len(bits)))


def _check_n(n: int) -> None:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")


def _as_qubit(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("qubits must be integers")
    return value


def _as_angle(value: object) -> float:
    if not isinstance(value, Real):
        raise TypeError("rotation angles must be real numbers")
    return float(value)


def _as_qubit_list(value: object) -> list[int]:
    if not isinstance(value, list):
        raise TypeError("the qubit collection must be a list")
    return [_as_qubit(item) for item in value]


def _as_matrix(value: object) -> ComplexMatrix:
    return np.asarray(cast(npt.ArrayLike, value), dtype=np.complex128)


def _check_qubits(n: int, qubits: list[int]) -> None:
    if len(set(qubits)) != len(qubits):
        raise ValueError("qubits must be distinct")
    if any(q < 0 or q >= n for q in qubits):
        raise ValueError(f"qubits must lie in range(0, {n})")


def _check_arity(op: Operation, arity: int) -> None:
    if len(op) != arity:
        raise ValueError(f"operation {op[0]!r} must have {arity} entries")


def _density_qubit_count(rho: ComplexMatrix) -> int:
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be a square matrix")
    dimension = rho.shape[0]
    n = 0
    power_of_two = 1
    while power_of_two < dimension:
        power_of_two *= 2
        n += 1
    if power_of_two != dimension:
        raise ValueError("rho dimension must be a power of two")
    return n


def embed(n: int, qubits: list[int], M: ComplexMatrix) -> ComplexMatrix:
    """Embed ``M`` on ``qubits`` using an explicit tensor product and permutation.

    ``M`` uses the tensor-factor order in ``qubits``.  First, the operator is built in
    the temporary order ``qubits + remaining_qubits``.  The permutation matrix maps each
    temporary-order basis vector to the corresponding canonical-order basis vector.
    """
    _check_n(n)
    _check_qubits(n, qubits)
    local = np.asarray(M, dtype=np.complex128)
    expected_dimension = 2 ** len(qubits)
    if local.shape != (expected_dimension, expected_dimension):
        raise ValueError(
            f"M has shape {local.shape}, expected {(expected_dimension, expected_dimension)}"
        )

    remaining = [q for q in range(n) if q not in qubits]
    temporary_order = qubits + remaining

    # This is literally M ⊗ I ⊗ ... ⊗ I in the temporary tensor-factor order.
    temporary_operator = local
    for _ in remaining:
        temporary_operator = np.asarray(np.kron(temporary_operator, _I), dtype=np.complex128)

    dimension = 2**n
    permutation = np.zeros((dimension, dimension), dtype=np.complex128)
    for temporary_index in range(dimension):
        temporary_bits = _basis_bits(temporary_index, n)
        canonical_bits = tuple(temporary_bits[temporary_order.index(q)] for q in range(n))
        canonical_index = _basis_index(canonical_bits)
        permutation[canonical_index, temporary_index] = 1.0

    return permutation @ temporary_operator @ permutation.conj().T


def _rx(theta: float) -> ComplexMatrix:
    cosine = np.cos(theta / 2.0)
    sine = np.sin(theta / 2.0)
    return np.array([[cosine, -1j * sine], [-1j * sine, cosine]], dtype=np.complex128)


def _ry(theta: float) -> ComplexMatrix:
    cosine = np.cos(theta / 2.0)
    sine = np.sin(theta / 2.0)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=np.complex128)


def _rz(theta: float) -> ComplexMatrix:
    return np.diag(np.array([np.exp(-0.5j * theta), np.exp(0.5j * theta)], dtype=np.complex128))


def _fixed_gate(name: str) -> ComplexMatrix:
    # Gate matrices use the standard convention requested for this oracle. [TASK SPECIFICATION]
    if name == "h":
        return np.asarray(
            np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) * 2.0**-0.5,
            dtype=np.complex128,
        )
    if name == "x":
        return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    if name == "y":
        return np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
    if name == "z":
        return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    if name == "s":
        return np.diag(np.array([1.0, 1j], dtype=np.complex128))
    if name == "t":
        return np.diag(np.array([1.0, np.exp(0.25j * np.pi)], dtype=np.complex128))
    raise ValueError(f"unknown fixed gate {name!r}")


def _apply_unitary(
    rho: ComplexMatrix, n: int, qubits: list[int], unitary: ComplexMatrix
) -> ComplexMatrix:
    full_unitary = embed(n, qubits, unitary)
    return full_unitary @ rho @ full_unitary.conj().T


def _apply_kraus(
    rho: ComplexMatrix, n: int, qubits: list[int], kraus_operators: list[ComplexMatrix]
) -> ComplexMatrix:
    result = np.zeros_like(rho, dtype=np.complex128)
    for local_operator in kraus_operators:
        full_operator = embed(n, qubits, local_operator)
        result = result + full_operator @ rho @ full_operator.conj().T
    return result


def _initial_density(n: int, rho0: ComplexMatrix | None) -> ComplexMatrix:
    dimension = 2**n
    if rho0 is not None:
        rho = np.asarray(rho0, dtype=np.complex128)
        if rho.shape != (dimension, dimension):
            raise ValueError(f"rho0 has shape {rho.shape}, expected {(dimension, dimension)}")
        return rho.copy()

    zero = np.array([1.0, 0.0], dtype=np.complex128)
    state = np.array([1.0], dtype=np.complex128)
    for _ in range(n):
        state = np.asarray(np.kron(state, zero), dtype=np.complex128)
    return np.outer(state, state.conj())


def run(n: int, ops: list[Operation], rho0: ComplexMatrix | None = None) -> ComplexMatrix:
    """Apply ``ops`` in order and return the resulting complex128 density matrix."""
    _check_n(n)
    rho = _initial_density(n, rho0)

    for op in ops:
        if not op or not isinstance(op[0], str):
            raise ValueError("each operation must begin with a string name")
        name = op[0]

        if name in {"h", "x", "y", "z", "s", "t"}:
            _check_arity(op, 2)
            q = _as_qubit(op[1])
            _check_qubits(n, [q])
            rho = _apply_unitary(rho, n, [q], _fixed_gate(name))
            continue

        if name in {"rx", "ry", "rz"}:
            _check_arity(op, 3)
            q = _as_qubit(op[1])
            theta = _as_angle(op[2])
            _check_qubits(n, [q])
            rotation = {"rx": _rx, "ry": _ry, "rz": _rz}[name](theta)
            rho = _apply_unitary(rho, n, [q], rotation)
            continue

        if name in {"crx", "cx", "cz"}:
            expected_arity = 4 if name == "crx" else 3
            _check_arity(op, expected_arity)
            control = _as_qubit(op[1])
            target = _as_qubit(op[2])
            _check_qubits(n, [control, target])
            if name == "crx":
                target_gate = _rx(_as_angle(op[3]))
            elif name == "cx":
                target_gate = _fixed_gate("x")
            else:
                target_gate = _fixed_gate("z")
            controlled = np.kron(_P0, _I) + np.kron(_P1, target_gate)
            rho = _apply_unitary(rho, n, [control, target], controlled)
            continue

        if name == "swap":
            _check_arity(op, 3)
            first = _as_qubit(op[1])
            second = _as_qubit(op[2])
            _check_qubits(n, [first, second])
            swap = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.complex128,
            )
            rho = _apply_unitary(rho, n, [first, second], swap)
            continue

        if name == "u":
            _check_arity(op, 3)
            qubits = _as_qubit_list(op[1])
            _check_qubits(n, qubits)
            unitary = _as_matrix(op[2])
            local_dimension = 2 ** len(qubits)
            if unitary.shape != (local_dimension, local_dimension):
                raise ValueError("unitary shape does not match its qubit list")
            rho = _apply_unitary(rho, n, qubits, unitary)
            continue

        if name == "kraus":
            _check_arity(op, 3)
            qubits = _as_qubit_list(op[1])
            _check_qubits(n, qubits)
            raw_operators = op[2]
            if not isinstance(raw_operators, list) or not raw_operators:
                raise ValueError("a Kraus operation needs a nonempty list of operators")
            kraus_operators = [_as_matrix(operator) for operator in raw_operators]
            local_dimension = 2 ** len(qubits)
            if any(
                operator.shape != (local_dimension, local_dimension) for operator in kraus_operators
            ):
                raise ValueError("Kraus operator shape does not match its qubit list")
            rho = _apply_kraus(rho, n, qubits, kraus_operators)
            continue

        if name == "reset":
            _check_arity(op, 2)
            q = _as_qubit(op[1])
            _check_qubits(n, [q])
            # {|0><0|, |0><1|} gives |0><0|_q tensor Tr_q(rho). [TASK SPECIFICATION]
            reset_zero = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
            reset_one = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
            rho = _apply_kraus(rho, n, [q], [reset_zero, reset_one])
            continue

        raise ValueError(f"unsupported operation {name!r}")

    return np.asarray(rho, dtype=np.complex128)


def probabilities(rho: ComplexMatrix) -> RealVector:
    """Return computational-basis probabilities in canonical msb-first order."""
    matrix = np.asarray(rho, dtype=np.complex128)
    _density_qubit_count(matrix)
    return np.asarray(np.real(np.diag(matrix)), dtype=np.float64)


def marginal(rho: ComplexMatrix, keep: list[int]) -> ComplexMatrix:
    """Trace out unkept qubits, retaining tensor factors in exactly ``keep`` order."""
    matrix = np.asarray(rho, dtype=np.complex128)
    n = _density_qubit_count(matrix)
    _check_qubits(n, keep)
    traced = [q for q in range(n) if q not in keep]

    kept_dimension = 2 ** len(keep)
    traced_dimension = 2 ** len(traced)
    reduced = np.zeros((kept_dimension, kept_dimension), dtype=np.complex128)

    for kept_row_index in range(kept_dimension):
        kept_row_bits = _basis_bits(kept_row_index, len(keep))
        for kept_column_index in range(kept_dimension):
            kept_column_bits = _basis_bits(kept_column_index, len(keep))
            for traced_index in range(traced_dimension):
                traced_bits = _basis_bits(traced_index, len(traced))
                canonical_row_bits = tuple(
                    kept_row_bits[keep.index(q)] if q in keep else traced_bits[traced.index(q)]
                    for q in range(n)
                )
                canonical_column_bits = tuple(
                    kept_column_bits[keep.index(q)] if q in keep else traced_bits[traced.index(q)]
                    for q in range(n)
                )
                canonical_row = _basis_index(canonical_row_bits)
                canonical_column = _basis_index(canonical_column_bits)
                reduced[kept_row_index, kept_column_index] += matrix[
                    canonical_row, canonical_column
                ]

    return reduced
