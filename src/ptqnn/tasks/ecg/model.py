"""Encoding, forward pass and reconstruction metrics for ECG denoising ``[PTQNN §III.D]``.

The architecture is the one of ``[PTQNN Fig 12]``: five input qubits carrying the window
``[s(t−2) … s(t+2)]``, one to three prototype (or classic) hidden neurons, and one classic
output neuron whose activation ``P(out = 1)`` *is* the denoised sample ``ŝ(t)`` — Fig 12 draws
the arrow from the output wire straight into the blue denoised trace, and the network has no
other read-out.  Nothing is rescaled on the way out: ``P(out = 1) ∈ [0, 1]`` and the signal
lives in ``[0, 1]``.

Metrics are Eq 6–8, implemented literally; see :func:`psnr` and :func:`ssim` for the two places
where the printed equations need interpretation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt

from ptqnn.backends.base import Backend
from ptqnn.circuits import ir
from ptqnn.circuits.neurons import Network
from ptqnn.train.distribution import output_activation

Real = npt.NDArray[np.float64]
Complex = npt.NDArray[np.complex128]

# Value → angle encoding.  The prerequisite paper pins it numerically: ``[SNN Fig 4, §III.A]``
# draws the two-input perceptron with the numbers filled in — ``CRX(1.369)`` annotated ``[0.4]``
# and ``CRX(1.7722)`` annotated ``[0.6]`` — and ``2·asin(√0.4) = 1.3694``, ``2·asin(√0.6) =
# 1.7722``.  With ``RX(θ) = exp(−iθX/2)`` that is ``P(|1⟩) = sin²(θ/2) = s``: the amplitude *is*
# the value (docs/physics/rosenhahn_osborne_hirche_snn.md §3.3, verified to five digits).
#
# ``[PTQNN Fig 12]`` cannot overrule this.  Its five printed input angles (0.54454, 0.78147,
# 0.51643, 0.70226, 0.78199) are inconsistent with *any* monotone encoding of the five signal
# values read off the window it highlights — the circuit screenshot is of a different window
# (docs/physics/rosenhahn_hirche_ptqnn.md §6.4).  So "amplitude" is the default and the linear
# family ``θ = scale·s`` (with ``scale = π`` the earlier reading θ = π·s) is kept as an ablation;
# ``experiments/ecg_table1.py --encoding linear`` runs the whole grid under it.
#
# [ASSUMPTION] Noisy samples leaving [0, 1] are clipped before the arcsine; the encoding has no
# meaning outside the unit interval.  At σ = 0.02 the clean signal's floor of 0.117 is 5.8 σ
# from 0, so this never fires in practice.
Encoding = Literal["amplitude", "linear"]
DEFAULT_ENCODING: Encoding = "amplitude"
DEFAULT_SCALE = np.pi


def angles(
    window: Sequence[float] | Real,
    encoding: Encoding = DEFAULT_ENCODING,
    scale: float = DEFAULT_SCALE,
) -> Real:
    """Input rotation angles for one window (or, elementwise, for a whole batch)."""
    w = np.asarray(window, dtype=np.float64)
    if encoding == "amplitude":
        return 2.0 * np.arcsin(np.sqrt(np.clip(w, 0.0, 1.0)))
    return float(scale) * w


def encode(
    window: Sequence[float] | Real,
    encoding: Encoding = DEFAULT_ENCODING,
    scale: float = DEFAULT_SCALE,
) -> list[ir.Op]:
    """Input preparation of ``[PTQNN Fig 12]``: one ``RX`` per input qubit."""
    return [ir.rx(i, float(a)) for i, a in enumerate(angles(np.ravel(window), encoding, scale))]


def decode(
    theta: Sequence[float] | Real,
    encoding: Encoding = DEFAULT_ENCODING,
    scale: float = DEFAULT_SCALE,
) -> Real:
    """Inverse of :func:`angles` — used to check printed figure angles against the signal."""
    a = np.asarray(theta, dtype=np.float64)
    if encoding == "amplitude":
        return np.asarray(np.sin(a / 2.0) ** 2, dtype=np.float64)
    return a / float(scale)


# --- forward pass -----------------------------------------------------------------------------


def _mix(view: Complex, m: Complex, axis: int) -> None:
    """In-place 2×2 mixing of ``view`` along ``axis`` (a length-2 axis): the whole cost of the
    forward pass.  Written as slice arithmetic rather than ``tensordot`` because the batched
    state is tens of megabytes and every avoided copy shows up in the training time."""
    s0: list[slice | int] = [slice(None)] * view.ndim
    s1: list[slice | int] = [slice(None)] * view.ndim
    s0[axis], s1[axis] = 0, 1
    i0, i1 = tuple(s0), tuple(s1)
    a0 = view[i0].copy()
    a1 = view[i1]
    view[i0] = m[0, 0] * a0 + m[0, 1] * a1
    view[i1] = m[1, 0] * a0 + m[1, 1] * a1


def _apply_gate_batched(state: Complex, gate: ir.Gate, n: int) -> None:
    """Apply ``gate`` in place to a ``(B, 2^n)`` batch of statevectors.

    Handles the only two shapes this task builds: a single-qubit gate, and a controlled
    single-qubit gate (``|0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ U``, the form :func:`ptqnn.circuits.ir.crx`
    emits).  Anything else raises rather than silently taking a slow path.
    """
    B = state.shape[0]
    if gate.n_qubits == 1:
        (q,) = gate.qubits
        _mix(state.reshape(B, 2**q, 2, 2 ** (n - q - 1)), gate.matrix, 2)
        return
    if gate.n_qubits != 2:
        raise NotImplementedError(f"gate {gate.name!r} acts on {gate.n_qubits} qubits")
    c, tq = gate.qubits
    u = gate.matrix[2:, 2:]
    if not np.allclose(gate.matrix[:2, :2], np.eye(2)) or np.any(gate.matrix[:2, 2:]):
        raise NotImplementedError(f"gate {gate.name!r} is not a controlled single-qubit gate")
    lo, hi = min(c, tq), max(c, tq)
    view = state.reshape(B, 2**lo, 2, 2 ** (hi - lo - 1), 2, 2 ** (n - hi - 1))
    if c < tq:
        _mix(view[:, :, 1], u, 3)
    else:
        _mix(view[:, :, :, :, 1], u, 2)


def _batched_activation(
    net: Network, params: Real, X: Real, encoding: Encoding, scale: float
) -> Real:
    """``P(out = 1)`` for a whole batch of windows in one statevector evolution.

    Every window runs the *same* gate sequence and differs only in the product input state, so
    the batch is carried as an extra leading axis of the statevector: memory ``B · 2^n``, and one
    pass per gate instead of one per gate *and* window.  For 1800 windows on 9 qubits this is
    ~100× faster than looping the reference backend, which is what makes a KW–SA sweep over the
    whole signal affordable.
    """
    n = net.n_qubits
    B = X.shape[0]
    # product state of the five inputs: RX(θ)|0⟩ = [cos(θ/2), −i sin(θ/2)]
    half = 0.5 * angles(X, encoding, scale)
    amps = np.stack([np.cos(half), -1j * np.sin(half)], axis=-1)  # (B, n_inputs, 2)
    v = np.ones((B, 1), dtype=np.complex128)
    for i in range(net.n_inputs):
        v = (v[:, :, None] * amps[:, i, None, :]).reshape(B, -1)
    state = np.zeros((B, 2**n), dtype=np.complex128)
    state[:, :: 2 ** (n - net.n_inputs)] = v  # hidden + output qubits start in |0⟩ (low bits)

    for op in net.circuit(params).ops:
        if not isinstance(op, ir.Gate):
            raise TypeError("the batched path is unitary-only; pass a backend for channels")
        _apply_gate_batched(state, op, n)
    p = np.abs(state) ** 2
    return np.asarray(p[:, 1::2].sum(axis=1), dtype=np.float64)  # output qubit is the LSB


def predict(
    net: Network,
    params: Real,
    X: Real,
    backend: Backend | None = None,
    encoding: Encoding = DEFAULT_ENCODING,
    scale: float = DEFAULT_SCALE,
) -> Real:
    """Denoised samples ``ŝ = P(out = 1)`` for every window (row) of ``X``.

    ``backend=None`` uses the batched statevector path above; passing a backend evaluates the
    windows one at a time through it (slow, but the oracle the batched path is tested against,
    and the only way to run a noisy backend).

    No memoisation over windows: the samples are continuous floats, so two windows of a noisy
    signal never coincide and a cache would only pay for its own bookkeeping.  (It would pay on
    a quantised signal — MIT-BIH is 11-bit — but only for windows repeated *exactly* in all five
    positions, which does not happen at 1800 samples either.)
    """
    Xa = np.asarray(X, dtype=np.float64)
    if Xa.ndim != 2 or Xa.shape[1] != net.n_inputs:
        raise ValueError(f"X must be (N, {net.n_inputs}), got {Xa.shape}")
    p = np.asarray(params, dtype=np.float64)
    if backend is None:
        return _batched_activation(net, p, Xa, encoding, scale)
    return np.array([output_activation(net, p, encode(x, encoding, scale), backend) for x in Xa])


# --- metrics ``[PTQNN Eq 6–8]`` -----------------------------------------------------------------


def mse(s1: Real, s2: Real) -> float:
    a, b = np.asarray(s1, dtype=np.float64), np.asarray(s2, dtype=np.float64)
    return float(np.mean((a - b) ** 2))


def psnr(s1: Real, s2: Real, R: float | None = None) -> float:
    """``PSNR = 10 log10(R² / MSE(s1, s2)) = 20 log10(R / √MSE)`` ``[PTQNN Eq 6, 7]``.

    "with MSE denoting the Mean Squared Error and R denoting the maximum fluctuation in the
    signal" ``[PTQNN §III.D]``.  [ASSUMPTION] "Maximum fluctuation" is read as the peak-to-peak
    range of the *reference* signal, ``R = max(s1) − min(s1)`` (≈ 0.80 for our ECG); the usual
    image-processing convention ``R = 1`` for data in [0, 1] would add a fixed
    ``20 log10(1/0.80) = 1.9`` dB to every row of Table I and is available by passing ``R=1``.
    Identical signals give ``+inf``.
    """
    m = mse(s1, s2)
    if m == 0.0:
        return float("inf")
    r = float(np.ptp(np.asarray(s1, dtype=np.float64))) if R is None else float(R)
    return float(10.0 * np.log10(r**2 / m))


def ssim(s1: Real, s2: Real, C1: float = 1e-4, C2: float = 9e-4) -> float:
    """Global (single-window) SSIM ``[PTQNN Eq 8]``.

    ``SSIM = (2 µ1 µ2 + C1)(2 σ12 + C2) / ((µ1² + µ2² + C1)(σ1² + σ2² + C2))``, "where µ and σ
    denote the mean, variance and covariance of the signals and C are (small) stabilizer
    constants to avoid division by zero" ``[PTQNN §III.D]``.  As printed, Eq 8's denominator
    reads ``(µ²_{s1} + µ²_{s1} + C1)(σ²_{s1} + σ²_{s1} + C2)`` — a typo in the paper, since it
    would make SSIM(s1, s2) a function of s1 alone in the denominator and break SSIM(s, s) = 1.
    We use the standard form with ``s2`` in the second term ``[Wang 2004, Eq 13]``.

    ``C1 = (0.01 L)² = 1e-4`` and ``C2 = (0.03 L)² = 9e-4`` for dynamic range ``L = 1``
    ``[ASSUMPTION]`` — the paper only calls them "(small) stabilizer constants".

    This is the *global* statistic literally written in Eq 8: one mean, variance and covariance
    over the whole signal.  scikit-image's ``structural_similarity`` instead averages the same
    expression over sliding local windows (Wang et al.'s actual proposal), which is a much
    harsher measure for a smooth signal — the global version saturates near 1 for anything that
    tracks the reference at all.  :func:`ssim_windowed` provides that variant, and the ECG
    experiment reports both.
    """
    a = np.asarray(s1, dtype=np.float64).ravel()
    b = np.asarray(s2, dtype=np.float64).ravel()
    mu1, mu2 = float(a.mean()), float(b.mean())
    v1, v2 = float(a.var()), float(b.var())
    cov = float(((a - mu1) * (b - mu2)).mean())
    num = (2 * mu1 * mu2 + C1) * (2 * cov + C2)
    den = (mu1**2 + mu2**2 + C1) * (v1 + v2 + C2)
    return float(num / den)


def ssim_windowed(s1: Real, s2: Real, win: int = 7, C1: float = 1e-4, C2: float = 9e-4) -> float:
    """Mean of Eq 8 over sliding windows of ``win`` samples — the scikit-image convention
    ``[Wang 2004 §III.B]`` ("the SSIM index is applied locally rather than globally ... the
    resulting SSIM index map ... we use a mean SSIM index to evaluate the overall image
    quality").  1-D analogue with a uniform (not Gaussian) window; reported next to
    :func:`ssim` because the paper's 0.87–0.93 are far below what the global statistic can
    produce for a signal this smooth.
    """
    a = np.asarray(s1, dtype=np.float64).ravel()
    b = np.asarray(s2, dtype=np.float64).ravel()
    if a.size < win:
        raise ValueError("signal shorter than the SSIM window")
    va = np.lib.stride_tricks.sliding_window_view(a, win)
    vb = np.lib.stride_tricks.sliding_window_view(b, win)
    mu1, mu2 = va.mean(axis=1), vb.mean(axis=1)
    v1, v2 = va.var(axis=1), vb.var(axis=1)
    cov = ((va - mu1[:, None]) * (vb - mu2[:, None])).mean(axis=1)
    num = (2 * mu1 * mu2 + C1) * (2 * cov + C2)
    den = (mu1**2 + mu2**2 + C1) * (v1 + v2 + C2)
    return float(np.mean(num / den))


# --- loss -------------------------------------------------------------------------------------


def mse_loss(
    net: Network,
    X: Real,
    y: Real,
    backend: Backend | None = None,
    subsample: int | None = None,
    seed: int = 0,
    encoding: Encoding = DEFAULT_ENCODING,
    scale: float = DEFAULT_SCALE,
) -> Callable[[Real], float]:
    """``mean((P(out=1) − s(t))²)`` over the windows — the objective handed to
    :func:`ptqnn.train.kwsa.minimise`.

    ``subsample`` draws a *fixed* random subset of the windows once, at construction: the KW
    finite differences compare ``f(x + c)`` with ``f(x − c)``, so a subset that changed between
    evaluations would turn the gradient estimate into pure sampling noise.  Cost per evaluation
    then scales with the subset, not with the signal length.
    """
    Xa = np.asarray(X, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if subsample is not None and subsample < Xa.shape[0]:
        idx = np.random.default_rng(seed).choice(Xa.shape[0], size=subsample, replace=False)
        idx.sort()
        Xa, ya = Xa[idx], ya[idx]

    def loss(params: Real) -> float:
        return float(np.mean((predict(net, params, Xa, backend, encoding, scale) - ya) ** 2))

    return loss
