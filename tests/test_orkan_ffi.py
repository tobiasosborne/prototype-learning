"""Contract tests for the raw orkan ctypes binding.

Everything orkan computes is checked against an *independent* numpy reference built
in this file out of `np.kron` and 2x2 blocks -- deliberately no shared index helper
with :mod:`ptqnn.backends._orkan_ffi`, because the whole point of these tests is to
pin down orkan's index conventions rather than to re-assert our own arithmetic.

The reference lives in the **msb0** convention

    basis index = sum_q b_q 2^(n-1-q)      (qubit 0 = most significant bit)

so that `kron(O_0, O_1, ..., O_{n-1})` is literally the operator that applies `O_q`
to qubit `q`.  orkan itself is lsb0 (see `test_bit_ordering_is_lsb0`); the binding's
``to_numpy(order="msb0")`` does the bit-reversal permutation, and the fact that the
two agree on every gate below is what certifies that permutation.
"""

from __future__ import annotations

import time
from functools import reduce

import numpy as np
import pytest

from ptqnn.backends import _orkan_ffi as ffi

pytestmark = [
    pytest.mark.orkan,
    pytest.mark.skipif(not ffi.orkan_available(), reason="liborkan.so not built/found"),
]

TOL = 1e-12

# ---------------------------------------------------------------------------
# Independent numpy reference (msb0: qubit 0 == most significant bit)
# ---------------------------------------------------------------------------

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.diag([1, 1j]).astype(complex)
SDG = S.conj().T
T = np.diag([1, np.exp(1j * np.pi / 4)]).astype(complex)
TDG = T.conj().T
P0 = np.array([[1, 0], [0, 0]], dtype=complex)
P1 = np.array([[0, 0], [0, 1]], dtype=complex)


def rx(t: float) -> np.ndarray:
    return np.cos(t / 2) * I2 - 1j * np.sin(t / 2) * X


def ry(t: float) -> np.ndarray:
    return np.cos(t / 2) * I2 - 1j * np.sin(t / 2) * Y


def rz(t: float) -> np.ndarray:
    return np.diag([np.exp(-1j * t / 2), np.exp(1j * t / 2)]).astype(complex)


def phase(t: float) -> np.ndarray:
    return np.diag([1.0, np.exp(1j * t)]).astype(complex)


def kron_all(mats: list[np.ndarray]) -> np.ndarray:
    """`kron(m[0], m[1], ...)` -- msb0 by construction."""
    return reduce(np.kron, mats)


def op1(n: int, q: int, u: np.ndarray) -> np.ndarray:
    """Embed a 2x2 `u` on qubit `q` of `n` (msb0)."""
    return kron_all([u if k == q else I2 for k in range(n)])


def op2(n: int, q0: int, q1: int, u4: np.ndarray) -> np.ndarray:
    """Embed a 4x4 `u4` whose basis index is ``2*b_q0 + b_q1`` (q0 = upper factor).

    Written as the 16-term sum ``sum_{ikjl} u4[2i+j, 2k+l] |i><k|_q0 (x) |j><l|_q1``
    so that it is built purely from single-qubit krons -- no index tricks.
    """
    out = np.zeros((2**n, 2**n), dtype=complex)
    for i in range(2):
        for k in range(2):
            e_ik = np.zeros((2, 2), dtype=complex)
            e_ik[i, k] = 1.0
            for j in range(2):
                for ll in range(2):
                    e_jl = np.zeros((2, 2), dtype=complex)
                    e_jl[j, ll] = 1.0
                    out += u4[2 * i + j, 2 * k + ll] * (op1(n, q0, e_ik) @ op1(n, q1, e_jl))
    return out


def ctrl(n: int, c: int, t: int, u: np.ndarray) -> np.ndarray:
    """Controlled-`u`, control `c`, target `t` (msb0), from krons only."""
    return op1(n, c, P0) + op1(n, c, P1) @ op1(n, t, u)


def ccx_ref(n: int, c1: int, c2: int, t: int) -> np.ndarray:
    both = op1(n, c1, P1) @ op1(n, c2, P1)
    return (np.eye(2**n, dtype=complex) - both) + both @ op1(n, t, X)


def swap_ref(n: int, a: int, b: int) -> np.ndarray:
    return ctrl(n, a, b, X) @ ctrl(n, b, a, X) @ ctrl(n, a, b, X)


def crx_ref(theta: float) -> np.ndarray:
    """CRX(theta) as a 4x4 in the ``2*b_q0 + b_q1`` basis, control = q0."""
    return np.kron(P0, I2) + np.kron(P1, rx(theta))


def haar_unitary(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(a)
    return q * (np.diag(r) / np.abs(np.diag(r)))


MIXED = [ffi.StateType.MIXED_TILED, ffi.StateType.MIXED_PACKED]
ALL_TYPES = [ffi.StateType.PURE, *MIXED]


def zero_vec(n: int) -> np.ndarray:
    v = np.zeros(2**n, dtype=complex)
    v[0] = 1.0
    return v


def dm(v: np.ndarray) -> np.ndarray:
    return np.outer(v, v.conj())


# ---------------------------------------------------------------------------
# Availability / library metadata
# ---------------------------------------------------------------------------


def test_library_loads_and_reports_tile_dim() -> None:
    lib = ffi.load_orkan()
    assert lib.tile_dim >= 8
    assert lib.tile_dim & (lib.tile_dim - 1) == 0  # power of two


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [3, 4])
@pytest.mark.parametrize("kind", ALL_TYPES)
def test_zero_init(n: int, kind: ffi.StateType) -> None:
    with ffi.OrkanState.zero(n, kind) as st:
        got = st.to_numpy(order="msb0")
        want = zero_vec(n) if kind is ffi.StateType.PURE else dm(zero_vec(n))
        assert np.allclose(got, want, atol=TOL)


@pytest.mark.parametrize("n", [3, 4])
@pytest.mark.parametrize("kind", ALL_TYPES)
def test_plus_init(n: int, kind: ffi.StateType) -> None:
    plus = np.full(2**n, 2.0 ** (-n / 2), dtype=complex)
    with ffi.OrkanState.plus(n, kind) as st:
        got = st.to_numpy(order="msb0")
        want = plus if kind is ffi.StateType.PURE else dm(plus)
        assert np.allclose(got, want, atol=TOL)


# ---------------------------------------------------------------------------
# Bit ordering -- the empirical determination
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [3, 4])
def test_bit_ordering_is_lsb0(n: int) -> None:
    """X on qubit q lands on *native* index 2**q => qubit 0 is the LSB."""
    for q in range(n):
        with ffi.OrkanState.zero(n, ffi.StateType.PURE) as st:
            st.x(q)
            native = st.to_numpy()
            (idx,) = np.nonzero(np.abs(native) > 0.5)
            assert idx[0] == 2**q


@pytest.mark.parametrize("n", [3, 4])
def test_msb0_is_bit_reversal_of_native(n: int) -> None:
    for q in range(n):
        with ffi.OrkanState.zero(n, ffi.StateType.PURE) as st:
            st.x(q)
            msb0 = st.to_numpy(order="msb0")
            (idx,) = np.nonzero(np.abs(msb0) > 0.5)
            assert idx[0] == 2 ** (n - 1 - q)


def test_msb0_permutation_matches_manual_bit_reversal() -> None:
    n = 4
    perm = ffi.msb0_permutation(n)
    for m in range(2**n):
        bits = [(m >> (n - 1 - q)) & 1 for q in range(n)]
        assert perm[m] == sum(b << q for q, b in enumerate(bits))


# ---------------------------------------------------------------------------
# Native gates
# ---------------------------------------------------------------------------

SINGLE_Q_GATES: list[tuple[str, tuple[float, ...], np.ndarray]] = [
    ("x", (), X),
    ("y", (), Y),
    ("z", (), Z),
    ("h", (), H),
    ("s", (), S),
    ("sdg", (), SDG),
    ("t", (), T),
    ("tdg", (), TDG),
    ("rx", (0.7,), rx(0.7)),
    ("ry", (-1.3,), ry(-1.3)),
    ("rz", (2.1,), rz(2.1)),
    ("p", (0.45,), phase(0.45)),
]


@pytest.mark.parametrize("kind", ALL_TYPES)
@pytest.mark.parametrize(("name", "args", "mat"), SINGLE_Q_GATES, ids=lambda a: str(a))
def test_single_qubit_gates(
    kind: ffi.StateType, name: str, args: tuple[float, ...], mat: np.ndarray
) -> None:
    n = 3
    for q in range(n):
        with ffi.OrkanState.zero(n, kind) as st:
            # scramble first so a wrong gate cannot hide on |0..0>
            for k in range(n):
                st.h(k)
            st.rz(0, 0.3)
            getattr(st, name)(q, *args)
            got = st.to_numpy(order="msb0")
        ref = op1(n, q, mat) @ op1(n, 0, rz(0.3)) @ kron_all([H] * n) @ zero_vec(n)
        want = ref if kind is ffi.StateType.PURE else dm(ref)
        assert np.allclose(got, want, atol=TOL), (kind, name, q)


@pytest.mark.parametrize("kind", ALL_TYPES)
def test_cx_cy_cz_swap(kind: ffi.StateType) -> None:
    n = 3
    for c in range(n):
        for t in range(n):
            if c == t:
                continue
            for name, u in (("cx", X), ("cy", Y), ("cz", Z)):
                with ffi.OrkanState.zero(n, kind) as st:
                    for k in range(n):
                        st.h(k)
                    st.rz(t, 0.9)
                    getattr(st, name)(c, t)
                    got = st.to_numpy(order="msb0")
                ref = ctrl(n, c, t, u) @ op1(n, t, rz(0.9)) @ kron_all([H] * n) @ zero_vec(n)
                want = ref if kind is ffi.StateType.PURE else dm(ref)
                assert np.allclose(got, want, atol=TOL), (kind, name, c, t)

            with ffi.OrkanState.zero(n, kind) as st:
                for k in range(n):
                    st.h(k)
                st.rz(t, 0.9)
                st.swap(c, t)
                got = st.to_numpy(order="msb0")
            ref = swap_ref(n, c, t) @ op1(n, t, rz(0.9)) @ kron_all([H] * n) @ zero_vec(n)
            want = ref if kind is ffi.StateType.PURE else dm(ref)
            assert np.allclose(got, want, atol=TOL), (kind, "swap", c, t)


@pytest.mark.parametrize("kind", ALL_TYPES)
def test_ccx(kind: ffi.StateType) -> None:
    n = 4
    for c1, c2, t in [(0, 1, 2), (2, 0, 3), (3, 1, 0)]:
        with ffi.OrkanState.zero(n, kind) as st:
            for k in range(n):
                st.h(k)
            st.rz(t, 0.6)
            st.ccx(c1, c2, t)
            got = st.to_numpy(order="msb0")
        ref = ccx_ref(n, c1, c2, t) @ op1(n, t, rz(0.6)) @ kron_all([H] * n) @ zero_vec(n)
        want = ref if kind is ffi.StateType.PURE else dm(ref)
        assert np.allclose(got, want, atol=TOL), (kind, c1, c2, t)


# ---------------------------------------------------------------------------
# Arbitrary unitaries via single_from_mat / two_from_mat  (MIXED_TILED only)
# ---------------------------------------------------------------------------


def _scrambled_tiled(n: int) -> ffi.OrkanState:
    st = ffi.OrkanState.zero(n, ffi.StateType.MIXED_TILED)
    for k in range(n):
        st.h(k)
        st.rz(k, 0.11 * (k + 1))
    st.cx(0, n - 1)
    return st


def _scrambled_ref(n: int) -> np.ndarray:
    v = zero_vec(n)
    for k in range(n):
        v = op1(n, k, rz(0.11 * (k + 1))) @ op1(n, k, H) @ v
    return ctrl(n, 0, n - 1, X) @ v


@pytest.mark.parametrize("n", [3, 4])
def test_apply_1q_unitary(n: int) -> None:
    u = haar_unitary(2, seed=11)
    for q in range(n):
        with _scrambled_tiled(n) as st:
            st.apply_1q_unitary(q, u)
            got = st.to_numpy(order="msb0")
        m = op1(n, q, u)
        want = m @ dm(_scrambled_ref(n)) @ m.conj().T
        assert np.allclose(got, want, atol=TOL), q


@pytest.mark.parametrize("n", [3, 4])
def test_apply_2q_unitary_random_both_orders(n: int) -> None:
    u = haar_unitary(4, seed=23)
    pairs = [(a, b) for a in range(n) for b in range(n) if a != b]
    for q0, q1 in pairs:
        with _scrambled_tiled(n) as st:
            st.apply_2q_unitary(q0, q1, u)
            got = st.to_numpy(order="msb0")
        m = op2(n, q0, q1, u)
        want = m @ dm(_scrambled_ref(n)) @ m.conj().T
        assert np.allclose(got, want, atol=TOL), (q0, q1)


@pytest.mark.parametrize("n", [3, 4])
def test_apply_2q_unitary_crx_both_orders(n: int) -> None:
    theta = 0.83
    u = crx_ref(theta)
    for q0, q1 in [(0, n - 1), (n - 1, 0), (1, 0), (0, 1)]:
        with _scrambled_tiled(n) as st:
            st.apply_2q_unitary(q0, q1, u)
            got = st.to_numpy(order="msb0")
        m = ctrl(n, q0, q1, rx(theta))
        want = m @ dm(_scrambled_ref(n)) @ m.conj().T
        assert np.allclose(got, want, atol=TOL), (q0, q1)


def test_apply_2q_unitary_matches_native_cx() -> None:
    """two_from_mat with the CX matrix must reproduce orkan's own `cx`."""
    n = 3
    cx4 = np.kron(P0, I2) + np.kron(P1, X)
    for c, t in [(0, 2), (2, 0), (1, 2)]:
        with _scrambled_tiled(n) as a, _scrambled_tiled(n) as b:
            a.apply_2q_unitary(c, t, cx4)
            b.cx(c, t)
            assert np.allclose(a.to_numpy(), b.to_numpy(), atol=TOL), (c, t)


def test_from_mat_rejects_non_tiled() -> None:
    for kind in (ffi.StateType.PURE, ffi.StateType.MIXED_PACKED):
        with ffi.OrkanState.zero(3, kind) as st:
            with pytest.raises(ffi.OrkanTypeError):
                st.apply_1q_unitary(0, X)
            with pytest.raises(ffi.OrkanTypeError):
                st.apply_2q_unitary(0, 1, np.eye(4, dtype=complex))


# ---------------------------------------------------------------------------
# Kraus channels
# ---------------------------------------------------------------------------


def amplitude_damping(gamma: float) -> list[np.ndarray]:
    k0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    k1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return [k0, k1]


def depolarising(p: float) -> list[np.ndarray]:
    return [
        np.sqrt(1 - 3 * p / 4) * I2,
        np.sqrt(p / 4) * X,
        np.sqrt(p / 4) * Y,
        np.sqrt(p / 4) * Z,
    ]


@pytest.mark.parametrize("kind", MIXED)
@pytest.mark.parametrize(
    ("label", "kraus"),
    [("amp_damp", amplitude_damping(0.31)), ("depol", depolarising(0.17))],
)
def test_kraus_1q(kind: ffi.StateType, label: str, kraus: list[np.ndarray]) -> None:
    n = 3
    ref0 = dm(_scrambled_ref(n))
    for q in range(n):
        st = ffi.OrkanState.zero(n, kind)
        with st:
            for k in range(n):
                st.h(k)
                st.rz(k, 0.11 * (k + 1))
            st.cx(0, n - 1)
            st.apply_kraus_1q(q, kraus)
            got = st.to_numpy(order="msb0")
        want = sum(op1(n, q, k) @ ref0 @ op1(n, q, k).conj().T for k in kraus)
        assert np.allclose(got, want, atol=TOL), (kind, label, q)
        assert abs(np.trace(got) - 1.0) < TOL
        assert np.allclose(got, got.conj().T, atol=TOL)


def test_kraus_rejects_pure() -> None:
    with ffi.OrkanState.zero(3, ffi.StateType.PURE) as st, pytest.raises(ffi.OrkanTypeError):
        st.apply_kraus_1q(0, amplitude_damping(0.1))


def test_kraus_rejects_non_cptp_shape() -> None:
    with ffi.OrkanState.zero(3, ffi.StateType.MIXED_TILED) as st:
        with pytest.raises(ValueError):
            st.apply_kraus_1q(0, [np.eye(4, dtype=complex)])
        with pytest.raises(ValueError):
            st.apply_kraus_1q(0, [])


# ---------------------------------------------------------------------------
# Guards and lifetime
# ---------------------------------------------------------------------------


def test_qubit_index_guards() -> None:
    with ffi.OrkanState.zero(3, ffi.StateType.MIXED_TILED) as st:
        with pytest.raises(IndexError):
            st.x(3)
        with pytest.raises(IndexError):
            st.x(-1)
        with pytest.raises(ValueError):
            st.cx(1, 1)
        with pytest.raises(ValueError):
            st.ccx(0, 1, 1)
        with pytest.raises(ValueError):
            st.apply_2q_unitary(0, 0, np.eye(4, dtype=complex))


def test_free_is_idempotent_and_use_after_free_raises() -> None:
    st = ffi.OrkanState.zero(3, ffi.StateType.PURE)
    st.free()
    st.free()
    with pytest.raises(RuntimeError):
        st.x(0)
    with pytest.raises(RuntimeError):
        st.to_numpy()


def test_context_manager_frees() -> None:
    with ffi.OrkanState.zero(3, ffi.StateType.MIXED_TILED) as st:
        assert st.n_qubits == 3
    assert st.freed


def test_no_leak_over_many_allocations() -> None:
    for _ in range(200):
        with ffi.OrkanState.zero(6, ffi.StateType.MIXED_TILED) as st:
            st.h(0)
            st.cx(0, 1)


# ---------------------------------------------------------------------------
# 12-qubit smoke test (timing reported)
# ---------------------------------------------------------------------------


def test_12_qubit_mixed_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    n = 12
    theta = 0.37
    t_alloc = time.perf_counter()
    st = ffi.OrkanState.zero(n, ffi.StateType.MIXED_TILED)
    t_alloc = time.perf_counter() - t_alloc
    with st:
        t0 = time.perf_counter()
        for q in range(n):
            st.h(q)
        t_h = time.perf_counter() - t0

        t0 = time.perf_counter()
        st.cx(0, 11)
        t_cx = time.perf_counter() - t0

        u = crx_ref(theta)
        t0 = time.perf_counter()
        st.apply_2q_unitary(0, 11, u)
        t_crx = time.perf_counter() - t0

        t0 = time.perf_counter()
        st.apply_kraus_1q(3, amplitude_damping(0.05))
        t_kraus = time.perf_counter() - t0

        tr = sum(st.get(i, i) for i in range(2**n))

    with capsys.disabled():
        print(
            f"\n[orkan 12q MIXED_TILED] alloc {t_alloc * 1e3:.1f} ms | "
            f"12x H {t_h * 1e3:.1f} ms | cx {t_cx * 1e3:.1f} ms | "
            f"CRX via two_from_mat {t_crx * 1e3:.1f} ms | "
            f"1q Kraus {t_kraus * 1e3:.1f} ms"
        )
    assert abs(tr.real - 1.0) < 1e-10
    assert abs(tr.imag) < 1e-10
