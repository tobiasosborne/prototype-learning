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
renormalised over *all* ``2^n_inputs`` input strings, and the loss is ``KL(target ‖ model)``
``[PTQNN Eq 4]``.  Unlike bit counting, the dataset codes are a small subset (14–21 of 4096) of
the input strings, and the prototype neuron's ``RX(φ)/CRX/RX(−φ)`` sandwich does *not* return the
input register to its initial state — the ``CRX`` entangles it — so probability can leak onto
non-dataset codes.  Renormalising over everything makes the loss pay for that leak, and this is
load-bearing:

* with the leak charged (``restrict=False``, the default) training drives it below 1 % on our
  splits, the un-computation then really does restore the input register, the model becomes a
  function of the single sample, and the held-out accuracy is 95 % — the paper's regime;
* post-selecting on the dataset codes instead (``restrict=True``) removes the pressure and the
  optimiser exploits interference *between* dataset samples: the training KL collapses to 1e-5
  and the training accuracy to the 98 % ceiling, but the leak grows to 24–70 % and preparing a
  single sample as a basis state gives chance-level (34 %) predictions.  A classifier that only
  works on the batch it was trained with is not a classifier.

The classic network is untouched by the choice: its neurons never act on the input qubits, so no
two dataset codes ever interfere and both variants coincide bit for bit.

``[PTQNN Fig 7]`` starts at KL ≈ 0.24 (classic) and ≈ 0.27 (prototype).  Under the full
renormalisation that fixes the initialisation scale: uniform angles in ±0.7 rad give a median
initial KL of 0.26 (classic) and 0.20 (prototype), while the ±π of
:meth:`~ptqnn.circuits.neurons.Network.random_params` gives 4.2 for the prototype net.  See
``ptqnn.tasks.iris.train.IrisConfig.init_scale`` — ``[ASSUMPTION]``, neither paper states it.

``[ASSUMPTION]`` read-out.  The paper says only "when applying the regressed values to the
nearest class labels".  The three target levels are known up to one global scale (they are in
the ratio 1 : 4 : 9), so we fit that scale to the regressed values without using any label
(:func:`fit_level_scale`) and then round to the nearest level.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from ptqnn.backends import batched
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
    merged = merge_duplicate_codes(codes, labels)
    w = (merged.labels + 1.0) ** 2
    p = np.zeros(2**n_bits, dtype=np.float64)
    p[merged.codes] = w / w.sum()
    return p


# --- fast exact evaluation ---------------------------------------------------------------------
#
# The output neuron is a classic perceptron ``[PTQNN Fig 4]``: ``RX(b)`` on the output qubit and
# one ``CRX(w_j)`` per hidden qubit.  All of these act on the *same* qubit and are diagonal in
# the hidden basis, so for a hidden configuration ``h`` the output sees a single rotation by
# ``b + Σ_j h_j w_j`` and ``P(out = 1 | x, h) = sin²((b + Σ_j h_j w_j)/2)``.  Simulating the
# output qubit is therefore unnecessary: we evolve the ``n_in + n_hid`` qubit register with the
# shared kernel :mod:`ptqnn.backends.batched` and contract.
# ``test_fast_evaluator_agrees_with_the_reference_backend`` pins this to the oracle.


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


def fast_joint(net: Network, params: npt.ArrayLike, psi0: Complex) -> Real:
    """Unnormalised ``P(inputs = x, out = 1)``, without simulating the output qubit."""
    psi_full = np.asarray(psi0, dtype=np.complex128).reshape(-1, 2)
    if not np.allclose(psi_full[:, 1], 0.0):
        raise ValueError("fast evaluator requires the output qubit prepared in |0⟩")
    n = net.n_inputs + net.n_hidden
    psi = psi_full[:, 0].reshape(1, -1)
    for gate in _hidden_ops(net, params):
        psi = batched.apply_gate_batched(psi, gate, n)
    p = (np.abs(psi[0]) ** 2).reshape(2**net.n_inputs, 2**net.n_hidden)
    return np.asarray(p @ output_activation_table(net, params), dtype=np.float64)


def _normalise(joint: Real) -> Real:
    z = joint.sum()
    return joint / z if z > 0 else np.full(joint.shape, 1.0 / joint.size)


def fast_model_distribution(net: Network, params: npt.ArrayLike, psi0: Complex) -> Real:
    """``P(inputs = x, out = 1)`` renormalised, without simulating the output qubit."""
    return _normalise(fast_joint(net, params, psi0))


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
    return _restrict(_normalise(joint), support)


# --- losses ------------------------------------------------------------------------------------


def kl_loss(
    net: Network,
    codes: Codes,
    labels: npt.ArrayLike,
    backend: Backend | None = None,
    restrict: bool = False,
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
    """The constant ``s`` with ``s·v(x) ≈ (label(x)+1)²`` — label-free calibration.

    The three target levels are known up to one global constant (they are in the ratio 1 : 4 : 9)
    which depends on the class composition of the evaluated set: with
    ``p(x) = (l_x+1)²/Σ_y (l_y+1)²`` the constant is ``s = Σ_y (l_y+1)²``, and the evaluated set
    is not the one the model was trained on.  We fit it *in log space*: minimise
    ``Σ_x min_l (log v(x) + log s − log l)²`` over ``log s``, a 1-D shift against the three fixed
    anchors ``log 1, log 4, log 9``.

    Least squares on the linear residual ``Σ_x min_l (s v(x) − l)²`` must not be used — shrinking
    ``s`` shrinks every residual, so it collapses onto "everything is class 0" once the values
    carry more than ~20 % noise; nor does the fixed point ``s ← Σ_x nearest level`` hold, it
    spirals down for the same reason
    (``test_level_scale_fit_survives_noisy_regressed_values``).
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0 or v.max() <= 0:
        return 1.0
    lv = np.log(np.maximum(v / v.sum(), 1e-15))
    anchors = np.log(LEVELS)
    lo = float(anchors.min() - lv.max()) - 1.0
    hi = float(anchors.max() - lv.min()) + 1.0
    grid = np.linspace(lo, hi, 4001)
    d = (lv[None, :, None] + grid[:, None, None] - anchors[None, None, :]) ** 2
    return float(np.exp(grid[int(d.min(axis=2).sum(axis=1).argmin())]))


def labels_from_values(values: Real, scale: float | None = None) -> npt.NDArray[np.int64]:
    """ "Applying the regressed values to the nearest class labels" ``[PTQNN §III.C]``.

    "Nearest" is taken in log space, consistently with :func:`fit_level_scale`: the class
    boundaries are the geometric means √(1·4) = 2 and √(4·9) = 6 rather than the arithmetic
    2.5 and 6.5.  The regression error on a probability that spans a factor 9 is multiplicative,
    so an arithmetic boundary systematically pulls the top class down.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v / v.sum() if v.sum() > 0 else v
    s = fit_level_scale(v) if scale is None else scale
    lv = np.log(np.maximum(s * v, 1e-15))
    return np.abs(lv[:, None] - np.log(LEVELS)[None, :]).argmin(axis=1).astype(np.int64)


def regressed_values(
    net: Network, params: npt.ArrayLike, codes: Codes, backend: Backend | None = None
) -> tuple[Codes, Real]:
    """``(distinct codes, model probability on each)`` for the superposition over ``codes``."""
    uniq = np.unique(np.asarray(codes, dtype=np.int64))
    psi0 = dataset_superposition(uniq, net.n_qubits, net.input_qubits)
    return uniq, model_distribution(net, params, psi0, backend, uniq)[uniq]


def basis_activations(
    net: Network, params: npt.ArrayLike, codes: Codes, backend: Backend | None = None
) -> tuple[Codes, Real]:
    """``(distinct codes, P(out = 1 | x))`` with each code prepared as a *basis* state.

    The per-sample read-out.  For a classic hidden layer it is proportional to the superposition
    read-out of :func:`regressed_values` (the input register is never touched, so no two dataset
    codes interfere); for a prototype layer the ``RX/CRX/RX(−φ)`` sandwich makes the two differ,
    and only this one is a function of the single sample.
    """
    uniq = np.unique(np.asarray(codes, dtype=np.int64))
    out = np.empty(uniq.size, dtype=np.float64)
    for i, c in enumerate(uniq):
        psi = dataset_superposition(np.array([c], dtype=np.int64), net.n_qubits, net.input_qubits)
        if backend is None:
            out[i] = fast_joint(net, params, psi).sum()
        else:
            p = backend.probabilities(net.circuit(params, initial_state=psi))
            out[i] = ref.marginal(p, [net.output_qubit])[1]
    return uniq, out


def calibrated_labels(
    net: Network,
    params: npt.ArrayLike,
    calib_codes: Codes,
    codes: Codes,
    backend: Backend | None = None,
) -> tuple[Codes, npt.NDArray[np.int64]]:
    """``(distinct codes, label)``: per-sample read-out, level scale fitted on ``calib_codes``.

    :func:`labels_from_values` normalises and fits the scale on the *evaluated* set, so a
    code's label can change when other codes join or leave that set: in the 20-seed sweep
    (``iris_sweep/20260925T172101Z``) dropping 3 test codes moved the fitted scale of one
    prototype model by a class step, 94 % → 39 %.  Here the scale ``s`` and the normaliser
    ``Z = Σ_{x ∈ calib} P(out=1|x)`` are fixed on the training codes and applied to
    ``P(out=1|x)/Z`` of each evaluated code, so every label is a function of its own code only
    (an inductive classifier).  ``[ASSUMPTION]`` calibrating on the training set is our choice;
    ``[PTQNN §III.C]`` says only "applying the regressed values to the nearest class labels".
    With ``codes = calib_codes`` it equals ``labels_from_values`` of the per-sample read-out.
    """
    _, a_cal = basis_activations(net, params, calib_codes, backend)
    uniq, act = basis_activations(net, params, codes, backend)
    z = float(a_cal.sum())
    if z <= 0:
        return uniq, np.zeros(uniq.size, dtype=np.int64)
    s = fit_level_scale(a_cal / z)
    lv = np.log(np.maximum(s * act / z, 1e-15))
    return uniq, np.abs(lv[:, None] - np.log(LEVELS)[None, :]).argmin(axis=1).astype(np.int64)


def calibrated_accuracy(
    net: Network,
    params: npt.ArrayLike,
    calib_codes: Codes,
    codes: Codes,
    labels: npt.ArrayLike,
    backend: Backend | None = None,
) -> float:
    """Per-sample accuracy of :func:`calibrated_labels` (scale fitted on ``calib_codes``)."""
    c = np.asarray(codes, dtype=np.int64)
    uniq, lab = calibrated_labels(net, params, calib_codes, c, backend)
    return float(np.mean(lab[np.searchsorted(uniq, c)] == np.asarray(labels, dtype=np.int64)))


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
