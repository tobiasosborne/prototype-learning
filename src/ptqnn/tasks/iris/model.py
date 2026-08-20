"""Dataset superposition, amplitude-encoded target and read-out for Iris ``[PTQNN §III.C]``.

State preparation ``[PTQNN §III.C]``: "each binary encoding corresponds to a unique entry in a
2^12-dimensional vector space. Thus, 12 qubits can be brought in a superposition to represent
the entire dataset. Therefore, quantum state preparation can be used, as suggested in
[8, 20, 21]."  We load the resulting statevector directly into
:attr:`ptqnn.circuits.ir.Circuit.initial_state` — in the noiseless case this is exactly what any
exact preparation circuit (Möttönen, QRAM) would produce.

Target ``[PTQNN Fig 8]``: "Each bin is a data sample and the amplitude encodes the class label
(three class labels are available)" — so the *amplitude* is proportional to ``label + 1`` and the
measured probability to ``(label + 1)²``.  The bar heights of Fig 8 confirm the square: the
top-right (training ground truth) histogram has 15 bins at three levels with counts (3, 5, 7)
and heights 0.0116 / 0.0465 / >0.1, i.e. 1/86, 4/86, 9/86 with 86 = 3·1 + 5·4 + 7·9; the
bottom-right (test ground truth) has 30 bins with counts (7, 12, 11) and heights
0.0065 / 0.026 / 0.0584 = 1/154, 4/154, 9/154 with 154 = 7·1 + 12·4 + 11·9.  Bar heights read
off the figure at ±0.0005; the exact integer normalisations 1/86 and 1/154 leave no doubt about
the 1 : 4 : 9 ratio.

Model distribution: as for bit counting ``[PTQNN Fig 5]`` it is ``P(inputs = x, out = 1)``
renormalised over ``x``, and the loss is ``KL(target ‖ model)`` ``[PTQNN Eq 4]``.  Unlike bit
counting, the dataset codes are a *subset* (14–21 of 4096) of the input strings, and the
prototype neuron's ``RX(φ)/CRX/RX(−φ)`` sandwich does not return the input register to its
initial state — the CRX entangles it — so probability leaks onto non-dataset codes (95 % of it
at a random initialisation).  We renormalise over the dataset codes, i.e. we post-select on the
input register being a valid code, because that is what ``[PTQNN Fig 8]`` plots: the predicted
histogram (bottom left, 30 bars) sums to 1.00 ± 0.01 over the dataset bins alone.  It also
reproduces the ``[PTQNN Fig 7]`` starting values: over 12 seeds the median initial KL is 0.20
(classic) and 0.23 (prototype) against the figure's 0.24 and 0.27, whereas renormalising over
all 4096 strings gives 3.6 for the prototype net.  Pass ``support=None`` to switch it off.

``[ASSUMPTION]`` read-out.  The paper says only "when applying the regressed values to the
nearest class labels".  The three target levels are known up to one global scale (they are in
the ratio 1 : 4 : 9), so we fit that scale to the regressed values without using any label
(:func:`fit_level_scale`) and then round to the nearest level.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from ptqnn.backends import reference as ref
from ptqnn.backends.base import Backend
from ptqnn.circuits import ir
from ptqnn.circuits.neurons import Network, classic_neuron, prototype_neuron
from ptqnn.tasks.iris.data import N_BITS, Codes, merge_duplicate_codes
from ptqnn.train import losses

Real = npt.NDArray[np.float64]
Complex = npt.NDArray[np.complex128]
Loss = Callable[[Real], float]

LEVELS = np.array([1.0, 4.0, 9.0])
"""``(label + 1)²`` for the three Iris classes ``[PTQNN Fig 8]``."""


# --- state preparation and target -------------------------------------------------------------


def dataset_superposition(
    codes: Codes, n_qubits_total: int, input_qubits: Sequence[int]
) -> Complex:
    """Uniform superposition over the *distinct* codes on the input register ⊗ ``|0…0⟩`` else.

    Duplicated codes collapse onto one basis state — a quantum state has one amplitude per basis
    element (see :func:`ptqnn.tasks.iris.data.merge_duplicate_codes`).
    """
    uniq = np.unique(np.asarray(codes, dtype=np.int64))
    n_in = len(input_qubits)
    idx = np.zeros(uniq.size, dtype=np.int64)
    for j, q in enumerate(input_qubits):
        idx |= ((uniq >> (n_in - 1 - j)) & 1) << (n_qubits_total - 1 - q)
    psi = np.zeros(2**n_qubits_total, dtype=np.complex128)
    psi[idx] = 1.0 / np.sqrt(uniq.size)
    return psi


def target_distribution(codes: Codes, labels: npt.ArrayLike, n_bits: int = N_BITS) -> Real:
    """``p(x) ∝ (label(x) + 1)²`` on the dataset codes, 0 elsewhere ``[PTQNN Fig 8]``."""
    y = np.asarray(labels, dtype=np.float64)
    merged = merge_duplicate_codes(np.asarray(codes, dtype=np.int64), np.rint(y).astype(np.int64))
    if not np.allclose(y, np.rint(y)):  # already-merged fractional labels: keep them
        c = np.asarray(codes, dtype=np.int64)
        order = np.argsort(c)
        uniq_codes, uniq_labels = c[order], y[order]
    else:
        uniq_codes, uniq_labels = merged.codes, merged.labels
    w = (uniq_labels + 1.0) ** 2
    p = np.zeros(2**n_bits, dtype=np.float64)
    p[uniq_codes] = w / w.sum()
    return p


# --- fast exact evaluation ---------------------------------------------------------------------
#
# The output neuron is a classic perceptron ``[PTQNN Fig 4]``: ``RX(b)`` on the output qubit and
# one ``CRX(w_j)`` per hidden qubit.  All of these act on the *same* qubit and are diagonal in
# the hidden basis, so for a hidden configuration ``h`` the output sees a single rotation by
# ``b + Σ_j h_j w_j`` and ``P(out = 1 | x, h) = sin²((b + Σ_j h_j w_j)/2)``.  Simulating the
# output qubit is therefore unnecessary: we evolve the ``n_in + n_hid`` qubit register and
# contract.  ``test_fast_evaluator_agrees_with_the_reference_backend`` pins this to the oracle.


def _apply_2x2(view: Complex, axis: int, m: Complex) -> None:
    """In-place ``view ← (m ⊗ I) view`` along a length-2 ``axis``."""
    i0: list[slice | int] = [slice(None)] * view.ndim
    i1: list[slice | int] = [slice(None)] * view.ndim
    i0[axis], i1[axis] = 0, 1
    t0, t1 = tuple(i0), tuple(i1)
    a = view[t0].copy()
    b = view[t1]
    view[t0] = m[0, 0] * a + m[0, 1] * b
    view[t1] = m[1, 0] * a + m[1, 1] * b


def _apply_gate(psi: Complex, n: int, gate: ir.Gate) -> None:
    """In-place application of an ``rx`` or ``crx`` gate to a contiguous statevector."""
    if gate.name == "rx":
        (q,) = gate.qubits
        theta = gate.params[0]
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        m = np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
        _apply_2x2(psi.reshape(1 << q, 2, 1 << (n - q - 1)), 1, m)
        return
    if gate.name != "crx":
        raise ValueError(f"fast evaluator only knows rx/crx, got {gate.name!r}")
    ctrl, tgt = gate.qubits
    theta = gate.params[0]
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    m = np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
    lo, hi = (ctrl, tgt) if ctrl < tgt else (tgt, ctrl)
    block = psi.reshape(1 << lo, 2, 1 << (hi - lo - 1), 2, 1 << (n - hi - 1))
    if ctrl < tgt:
        _apply_2x2(block[:, 1], 2, m)  # control = 1 branch; target is the second split axis
    else:
        _apply_2x2(block[:, :, :, 1], 1, m)


def _hidden_ops(net: Network, params: npt.ArrayLike) -> list[ir.Gate]:
    """The hidden layer only — built by the very functions the network builder uses."""
    hidden, _ = net.unpack(params)
    ops: list[ir.Gate] = []
    for hq, (angles, last) in zip(net.hidden_qubits, hidden, strict=True):
        if net.hidden == "prototype":
            ops += prototype_neuron(net.input_qubits, hq, angles, last)
        else:
            ops += classic_neuron(net.input_qubits, hq, angles, last)
    return ops


def output_activation_table(net: Network, params: npt.ArrayLike) -> Real:
    """``sin²((b + Σ_j h_j w_j)/2)`` for every hidden configuration ``h`` (hidden qubit 0 = MSB)."""
    _, (w_out, b_out) = net.unpack(params)
    nh = net.n_hidden
    bits = (np.arange(2**nh, dtype=np.int64)[:, None] >> np.arange(nh - 1, -1, -1)) & 1
    return np.sin((b_out + bits @ w_out) / 2.0) ** 2


def _restrict(p: Real, support: Codes | None) -> Real:
    """Post-select the input register onto ``support`` (the dataset codes) and renormalise."""
    if support is None:
        return p
    q = np.zeros_like(p)
    q[support] = p[support]
    z = q.sum()
    return q / z if z > 0 else np.full(q.shape, 1.0 / q.size)


def fast_model_distribution(net: Network, params: npt.ArrayLike, psi0: Complex) -> Real:
    """``P(inputs = x, out = 1)`` renormalised, without simulating the output qubit."""
    psi_full = np.asarray(psi0, dtype=np.complex128).reshape(-1, 2)
    if not np.allclose(psi_full[:, 1], 0.0):
        raise ValueError("fast evaluator requires the output qubit prepared in |0⟩")
    n = net.n_inputs + net.n_hidden
    psi = np.ascontiguousarray(psi_full[:, 0])
    for gate in _hidden_ops(net, params):
        _apply_gate(psi, n, gate)
    p = (np.abs(psi) ** 2).reshape(2**net.n_inputs, 2**net.n_hidden)
    joint = p @ output_activation_table(net, params)
    z = joint.sum()
    return joint / z if z > 0 else np.full(joint.shape, 1.0 / joint.size)


def model_distribution(
    net: Network,
    params: npt.ArrayLike,
    psi0: Complex,
    backend: Backend | None = None,
    support: Codes | None = None,
) -> Real:
    """``P(x | out = 1)`` over the ``2^n_inputs`` input strings ``[PTQNN Eq 4, Fig 5]``.

    ``backend=None`` uses the exact reduced evaluator above; pass
    :class:`~ptqnn.backends.reference.StatevectorBackend` for the oracle path.  ``support``
    additionally post-selects the input register onto the dataset codes (see module docstring).
    """
    if backend is None:
        return _restrict(fast_model_distribution(net, params, psi0), support)
    p = backend.probabilities(net.circuit(params, initial_state=np.asarray(psi0)))
    joint = ref.marginal(p, [*net.input_qubits, net.output_qubit]).reshape(2**net.n_inputs, 2)[:, 1]
    z = joint.sum()
    return _restrict(joint / z if z > 0 else np.full(joint.shape, 1.0 / joint.size), support)


# --- losses ------------------------------------------------------------------------------------


def kl_loss(
    net: Network,
    codes: Codes,
    labels: npt.ArrayLike,
    backend: Backend | None = None,
    restrict: bool = True,
) -> Loss:
    """``KL(target ‖ model)`` for the dataset ``(codes, labels)`` prepared in superposition."""
    psi0 = dataset_superposition(codes, net.n_qubits, net.input_qubits)
    target = target_distribution(codes, labels)
    support = np.unique(np.asarray(codes, dtype=np.int64)) if restrict else None

    def loss(params: Real) -> float:
        q = model_distribution(net, params, psi0, backend, support)
        return losses.kl_divergence(target, q)

    return loss


def train_test_losses(
    net: Network,
    params: npt.ArrayLike,
    train_codes: Codes,
    train_labels: npt.ArrayLike,
    test_codes: Codes,
    test_labels: npt.ArrayLike,
    backend: Backend | None = None,
) -> tuple[float, float]:
    """The two curves of ``[PTQNN Fig 7]``: KL on the training and on the test superposition."""
    p = np.asarray(params, dtype=np.float64)
    return (
        kl_loss(net, train_codes, train_labels, backend)(p),
        kl_loss(net, test_codes, test_labels, backend)(p),
    )


# --- read-out ----------------------------------------------------------------------------------


def fit_level_scale(values: Real) -> float:
    """Scale ``s`` minimising ``Σ_x min_l (s·v(x) − (l+1)²)²`` — label-free calibration.

    The three target levels are known up to one global constant (ratio 1 : 4 : 9), which depends
    on the class composition of the evaluated set.  Fitting it uses no labels: it is the 1-D
    k-means of the regressed values with the centroid *ratios* frozen.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0 or v.max() <= 0:
        return 1.0
    best_s, best_r = 1.0, np.inf
    for level in LEVELS:
        s = float(level / v.max())
        for _ in range(50):
            assigned = LEVELS[np.abs(s * v[:, None] - LEVELS[None, :]).argmin(axis=1)]
            denom = float(v @ v)
            s_new = float(assigned @ v / denom) if denom > 0 else s
            if s_new == s:
                break
            s = s_new
        assigned = LEVELS[np.abs(s * v[:, None] - LEVELS[None, :]).argmin(axis=1)]
        r = float(np.sum((s * v - assigned) ** 2))
        if r < best_r:
            best_s, best_r = s, r
    return best_s


def labels_from_values(values: Real, scale: float | None = None) -> npt.NDArray[np.int64]:
    """ "Applying the regressed values to the nearest class labels" ``[PTQNN §III.C]``."""
    v = np.asarray(values, dtype=np.float64)
    s = fit_level_scale(v) if scale is None else scale
    return np.abs(s * v[:, None] - LEVELS[None, :]).argmin(axis=1).astype(np.int64)


def regressed_values(
    net: Network, params: npt.ArrayLike, codes: Codes, backend: Backend | None = None
) -> tuple[Codes, Real]:
    """``(distinct codes, model probability on each)`` for the superposition over ``codes``."""
    uniq = np.unique(np.asarray(codes, dtype=np.int64))
    psi0 = dataset_superposition(uniq, net.n_qubits, net.input_qubits)
    return uniq, model_distribution(net, params, psi0, backend, uniq)[uniq]


def predict_labels(
    net: Network, params: npt.ArrayLike, codes: Codes, backend: Backend | None = None
) -> npt.NDArray[np.int64]:
    """One predicted class per *distinct* code, ordered like ``np.unique(codes)``."""
    _, values = regressed_values(net, params, codes, backend)
    return labels_from_values(values)


def accuracy(
    net: Network,
    params: npt.ArrayLike,
    codes: Codes,
    labels: npt.ArrayLike,
    backend: Backend | None = None,
) -> float:
    """Per-sample accuracy: every sample inherits the prediction of its code ``[PTQNN §III.C]``."""
    c = np.asarray(codes, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int64)
    uniq, values = regressed_values(net, params, c, backend)
    per_code = labels_from_values(values)
    pred = per_code[np.searchsorted(uniq, c)]
    return float(np.mean(pred == y))
