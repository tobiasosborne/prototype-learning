"""Training the Iris models and the ingredients of ``[PTQNN Fig 7]`` and ``[PTQNN Fig 9]``.

``[PTQNN §III.C]``: "Figure 7 shows the train and test loss during optimization using (left) the
classical perceptron model and (right) the prototype model. While both models can explain the
training data very well, the prototype model (with the same amount of parameters) on the right
shows better generalization performance."  Fig 7 (left, classic) runs 15 epochs and ends with
train loss ≈0 while the test loss turns up from ≈0.030 to ≈0.045; Fig 7 (right, prototype) runs
26 epochs with train and test descending together to ≈0.010 / 0.020.  One "epoch" here is one
KW–SA sweep over all parameters (:mod:`ptqnn.train.kwsa`, ``mode="coordinate"``).

``[PTQNN Fig 9]``: "Left: TSNE-plot of the training data (red) and the weights of the hidden
layer (blue) when using the classical architecture. Right: ... the prototype layer ... Whereas
the hidden data reveal only limited information about the dataset, are spanning the weights of
the prototype layer much better the dataset."  To place hidden neurons in the same space as the
binary data we map their angles to per-input activation probabilities
(:func:`hidden_layer_weights_for_tsne`, ``as_pattern=True``).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
from sklearn.manifold import TSNE

from ptqnn.backends import batched
from ptqnn.backends import reference as ref
from ptqnn.backends.base import Backend
from ptqnn.circuits import ir
from ptqnn.circuits.neurons import HiddenKind, Network
from ptqnn.tasks.iris import data as iris_data
from ptqnn.tasks.iris import model as iris_model
from ptqnn.tasks.iris.data import Codes, code_bits
from ptqnn.train import kwsa

Real = npt.NDArray[np.float64]
Complex = npt.NDArray[np.complex128]

# [ASSUMPTION] KW–SA hyper-parameters.  Neither paper publishes a_n, c_n, T or k.  These are the
# bit-count settings (`ptqnn.tasks.bitcount.SUPERPOSITION_CONFIG`) re-tuned for the Iris loss
# scale (KL starts near 0.25 instead of 0.055); see experiments/runs/iris/*/meta.json.
DEFAULT_A0 = 2.0
DEFAULT_C0 = 0.1


@dataclass(frozen=True)
class IrisConfig:
    """Everything a Fig-7 run needs besides the data."""

    iterations: int = 40
    n_hidden: int = 4
    """``[ASSUMPTION]`` — ``[SNN Table I]`` lists Iris as "12 inp, 4 perc, 19 qubits"; the
    prototype paper does not repeat the number.  With one output neuron we use 12 + 4 + 1 = 17
    qubits."""
    init_scale: float = 0.7
    """``[ASSUMPTION]`` half-width of the uniform initialisation.  ``[PTQNN Fig 7]`` starts at
    KL ≈ 0.24 (classic) / 0.27 (prototype); ±0.7 rad reproduces that (median 0.26 / 0.20 over 10
    seeds) whereas the ±π default of ``Network.random_params`` gives 4.2 for the prototype net,
    whose ``RX(φ)`` rotations then scatter the input register out of the code space."""
    a0: float = DEFAULT_A0
    alpha: float = 0.3
    c0: float = DEFAULT_C0
    gamma: float = 0.101
    t0: float = 0.0
    cooling: float = 0.8
    k: float = 1.0
    mode: Literal["coordinate", "full"] = "coordinate"

    def kwsa(self, seed: int) -> kwsa.KWSAConfig:
        return kwsa.KWSAConfig(
            iterations=self.iterations,
            a0=self.a0,
            alpha=self.alpha,
            c0=self.c0,
            gamma=self.gamma,
            t0=self.t0,
            cooling=self.cooling,
            k=self.k,
            seed=seed,
            mode=self.mode,
        )


@dataclass
class IrisResult:
    """Per-sweep train/test KL ``[PTQNN Fig 7]`` plus the best parameters found."""

    params: Real
    train_loss: list[float] = field(default_factory=list)
    test_loss: list[float] = field(default_factory=list)
    evaluations: int = 0
    hidden: HiddenKind = "prototype"
    seed: int = 0


def network(cfg: IrisConfig, hidden: HiddenKind) -> Network:
    """12 input qubits ``[SNN §IV.B]``, ``cfg.n_hidden`` hidden, one output ``[PTQNN Fig 4]``."""
    return Network(n_inputs=12, n_hidden=cfg.n_hidden, hidden=hidden)


def train(
    net: Network,
    train_codes: Codes,
    train_labels: npt.ArrayLike,
    test_codes: Codes,
    test_labels: npt.ArrayLike,
    cfg: IrisConfig,
    seed: int = 0,
    backend: Backend | None = None,
) -> IrisResult:
    """KW–SA on ``KL(target ‖ model)`` of the training superposition, tracking the test KL.

    The test loss is *monitored only* — it never enters the objective.
    """
    loss_train = iris_model.kl_loss(net, train_codes, train_labels, backend)
    loss_test = iris_model.kl_loss(net, test_codes, test_labels, backend)
    x0 = net.random_params(np.random.default_rng(seed), scale=cfg.init_scale)
    test_hist = [loss_test(x0)]

    def record(_n: int, x: Real, _fx: float) -> None:
        test_hist.append(loss_test(x))

    res = kwsa.minimise(loss_train, x0, cfg.kwsa(seed), callback=record)
    return IrisResult(
        params=res.params,
        train_loss=list(res.history),
        test_loss=test_hist,
        evaluations=res.evaluations,
        hidden=net.hidden,
        seed=seed,
    )


def run_fig7(
    train_codes: Codes,
    train_labels: npt.ArrayLike,
    test_codes: Codes,
    test_labels: npt.ArrayLike,
    cfg: IrisConfig | None = None,
    seed: int = 0,
    backend: Backend | None = None,
) -> dict[HiddenKind, IrisResult]:
    """Both panels of ``[PTQNN Fig 7]``: the classic and the prototype net, same parameter count.

    ``Network.n_params`` is ``n_hidden·(n_inputs+1) + n_hidden + 1`` for both kinds — "the
    prototype model (with the same amount of parameters)" ``[PTQNN §III.C]``.
    """
    c = cfg or IrisConfig()
    out: dict[HiddenKind, IrisResult] = {}
    kinds: tuple[HiddenKind, ...] = ("classic", "prototype")
    for kind in kinds:
        out[kind] = train(
            network(c, kind), train_codes, train_labels, test_codes, test_labels, c, seed, backend
        )
    return out


def hidden_layer_weights_for_tsne(
    net: Network, params: npt.ArrayLike, as_pattern: bool = False
) -> Real:
    """``(n_hidden, n_inputs)`` — the per-input angles of each hidden neuron ``[PTQNN Fig 9]``.

    ``as_pattern=True`` maps them into the ``[0, 1]`` data cube so that TSNE sees neurons and
    binary samples in one space (``[ASSUMPTION]`` — Fig 9 does not say how the weights were
    embedded):

    * prototype: ``cos²(φ_i/2)`` — the probability that the ``CRX`` fires when input ``i`` is 1
      (``RX(φ)|1⟩`` has ``P(1) = cos²(φ/2)``), i.e. how strongly the neuron matches a set bit;
    * classic: ``sin²(w_i/2)`` — the activation input ``i`` alone contributes when it is 1
      (a 0-valued input contributes nothing, the asymmetry ``[PTQNN §III.B]`` complains about).
    """
    hidden, _ = net.unpack(params)
    raw = np.stack([np.asarray(angles, dtype=np.float64) for angles, _ in hidden])
    if not as_pattern:
        return raw
    if net.hidden == "prototype":
        return np.cos(raw / 2.0) ** 2
    return np.sin(raw / 2.0) ** 2


def tsne_embedding(
    codes: Codes, weights: Real, seed: int = 0, perplexity: float | None = None
) -> tuple[Real, Real]:
    """TSNE of the training codes (as 0/1 vectors) together with the hidden patterns.

    Both are embedded in *one* TSNE fit — Fig 9 plots them in a common plane.
    """
    data = code_bits(np.unique(np.asarray(codes, dtype=np.int64))).astype(np.float64)
    w = np.asarray(weights, dtype=np.float64)
    stacked = np.vstack([data, w])
    p = perplexity if perplexity is not None else min(10.0, max(2.0, (stacked.shape[0] - 1) / 3))
    emb = TSNE(n_components=2, perplexity=p, random_state=seed, init="pca").fit_transform(stacked)
    emb = np.asarray(emb, dtype=np.float64)
    return emb[: data.shape[0]], emb[data.shape[0] :]


# --- quantitative prototype-placement metric (replaces the Fig 9 t-SNE as evidence) ---
#
# [PTQNN Fig 9] claims the prototype hidden weights "span" the dataset while the classic ones
# do not, but a t-SNE picture is eyeball evidence only: the embedding is stochastic, its
# perplexity is arbitrary, and proximity in it is not a distance statement about the 12-bit
# codes ([session-4 analysis]; the co-author asked for "another metric", e-mail 2026-08-27).
# The functions below reduce the claim to one number: how well the DECODED hidden patterns
# (one-hot per feature block, i.e. valid Iris codes) cover the distinct training codes.

Encoding = Literal["inverse", "direct"]
"""Which angle→probability map decodes a neuron to a pattern.

"inverse" is the ``[Rosenhahn e-mail 2026-08-21]`` encoding — a LARGE angle stresses a 0 input —
and coincides with ``hidden_layer_weights_for_tsne(as_pattern=True)``: cos²(φ/2) for prototype,
sin²(w/2) for classic.  "direct" is the complementary map: sin²(φ/2) / cos²(w/2).
"""


def block_argmax(patterns: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    """``(n, 12)`` non-negative scores → one-hot-per-block codes: per 3-bit feature block the
    argmax becomes 1, the others 0 (a valid ``[SNN §IV.B]`` Iris code).

    Invariant under any monotone rescaling of the input — only the per-block ordering is read
    (``argmax(m) == argmax(m**3)`` for ``m ≥ 0``) — so the metric does not care about the
    absolute probability scale, only which cluster wins.
    """
    m = np.asarray(patterns, dtype=np.float64)
    out = np.zeros(m.shape, dtype=np.uint8)
    rows = np.arange(m.shape[0])
    for f in range(iris_data.N_FEATURES):
        lo, hi = iris_data.K * f, iris_data.K * (f + 1)
        out[:, lo:hi][rows, m[:, lo:hi].argmax(axis=1)] = 1
    return out


def decode_hidden_patterns(
    net: Network, params: npt.ArrayLike, encoding: Encoding = "inverse"
) -> npt.NDArray[np.uint8]:
    """``(n_hidden, 12)`` one-hot patterns, one per hidden neuron.

    Per input ``i`` the score is ``m_i = cos²(φ_i/2)`` (prototype, "inverse") — the probability
    the ``CRX`` fires when input ``i`` is 1 ``[PTQNN Fig 2]``; the complementary map for
    "direct".  Classic neurons use ``sin²(w_i/2)`` ("inverse") / ``cos²(w_i/2)`` ("direct")
    ``[PTQNN Fig 1]``.  A large ``m_i`` in a block means the neuron is tuned to that cluster
    ("inverse": a large angle stresses a 0 input, ``[Rosenhahn e-mail 2026-08-21]``); the
    per-block argmax selects it.  Invariant under monotone rescaling of ``m`` — see
    :func:`block_argmax`.
    """
    hidden, _ = net.unpack(params)
    raw = np.stack([np.asarray(angles, dtype=np.float64) for angles, _ in hidden])
    if net.hidden == "prototype":
        m = np.cos(raw / 2.0) ** 2 if encoding == "inverse" else np.sin(raw / 2.0) ** 2
    else:
        m = np.sin(raw / 2.0) ** 2 if encoding == "inverse" else np.cos(raw / 2.0) ** 2
    return block_argmax(m)


def output_drive(net: Network, params: npt.ArrayLike) -> Real:
    """``(n_hidden,)`` — ``|sin(θ_j/2)·sin(w_j/2)|``, how strongly neuron ``j`` reaches the output.

    ``θ_j`` is the neuron's CRX angle and ``w_j`` its output-layer weight (``Network.unpack``
    gives both for the prototype kind: the shared CRX angle ``θ_j`` is the per-neuron scalar).
    """
    hidden, (w_out, _) = net.unpack(params)
    drive = np.zeros(net.n_hidden, dtype=np.float64)
    for j, (angles, scalar) in enumerate(hidden):
        if net.hidden == "prototype":
            d = abs(np.sin(scalar / 2.0))  # θ_j, the shared CRX angle [PTQNN Fig 2]
        else:
            # [ASSUMPTION] a classic neuron has no single CRX angle — its per-neuron scalar is
            # the bias [PTQNN Fig 1], and a bias offset does not decouple the neuron.  We take
            # the strongest input coupling, max_i |sin(w_i/2)|, as the drive instead.
            d = float(np.max(np.abs(np.sin(np.asarray(angles, dtype=np.float64) / 2.0))))
        drive[j] = d * abs(np.sin(w_out[j] / 2.0))
    return drive


def dead_hidden_neurons(
    net: Network, params: npt.ArrayLike, tol_rel: float = 0.1, tol_abs: float = 0.0
) -> npt.NDArray[np.bool_]:
    """``(n_hidden,)`` bool — neurons whose output coupling is negligible *relative to the layer*.

    Neuron ``j`` is dead when ``drive_j < tol_rel · max_k drive_k`` (or, with an absolute floor,
    when ``drive_j < tol_abs``), with ``drive`` from :func:`output_drive`.

    The tolerance is relative because the Iris target is *amplitude-encoded*
    (``p(x) ∝ (label+1)²`` ``[PTQNN Fig 8]``): the whole output distribution has to fit into
    three levels in the ratio 1 : 4 : 9 over ~13 codes, so the trained couplings are small in
    absolute terms — on ``experiments/runs/iris/20260918T085641Z-drop`` the prototype net has
    ``θ_j ∈ {0.45, −0.35, 0.31, −0.39}`` and ``w_out ∈ {0.16, −0.03, −0.56, −0.31}``, i.e.
    ``drive ≤ 0.06``.  The former absolute ``tol = 0.05`` therefore called all four neurons dead
    although they differ by a factor 20 among themselves; "dead" can only mean "dead compared
    with its neighbours".  ``[ASSUMPTION]`` ``tol_rel = 0.1`` — one order of magnitude below the
    strongest neuron; ``tol_abs = 0.0`` (off) by default.

    By construction the strongest neuron is never flagged unless ``tol_abs`` says so.
    """
    drive = output_drive(net, params)
    strongest = float(drive.max()) if drive.size else 0.0
    return np.asarray((drive < tol_rel * strongest) | (drive < tol_abs), dtype=np.bool_)


def covering_cost(patterns: npt.ArrayLike, codes: Codes) -> float:
    """Mean halved Hamming distance from each DISTINCT code to its nearest pattern.

    ``R = (1/M) Σ_{c ∈ C} min_j d_H(b_j, c)/2`` with ``C`` the distinct codes in ``codes``
    (``M`` of them) and ``b_j`` the decoded patterns.  One wrong 3-bit block costs 2 bits, so
    for one-hot patterns ``R ∈ [0, 4]`` (enforced: every pattern must be one-hot per block).
    """
    p = np.asarray(patterns, dtype=np.uint8)
    if p.ndim != 2 or p.shape[1] != iris_data.N_BITS:
        raise ValueError(f"patterns must be (k, {iris_data.N_BITS}), got {p.shape}")
    if not np.array_equal(
        p.reshape(-1, iris_data.N_FEATURES, iris_data.K).sum(axis=2),
        np.ones((p.shape[0], iris_data.N_FEATURES), dtype=np.uint8),
    ):
        raise ValueError("every pattern must contain exactly one 1-bit per 3-bit feature block")
    bits = iris_data.code_bits(np.unique(np.asarray(codes, dtype=np.int64)))
    d = (bits[:, None, :] != p[None, :, :]).sum(axis=2)  # (M, k) Hamming distances
    return float((d.min(axis=1) / 2.0).mean())


def valid_one_hot_patterns() -> npt.NDArray[np.uint8]:
    """All 81 valid Iris codes: 3 one-hot cluster choices per feature ``[SNN §IV.B]``."""
    combos = itertools.product(range(iris_data.K), repeat=iris_data.N_FEATURES)
    pats = np.zeros((iris_data.K**iris_data.N_FEATURES, iris_data.N_BITS), dtype=np.uint8)
    for i, clusters in enumerate(combos):
        for f, c in enumerate(clusters):
            pats[i, iris_data.K * f + c] = 1
    return pats


def covering_null(
    codes: Codes, k: int, n_draws: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    """``(mean, std)`` of ``R`` over ``k``-subsets drawn uniformly without replacement from the
    81 valid one-hot patterns, scored against the same distinct code set — the "the weights are
    no better than random patterns" baseline for ``[PTQNN Fig 9]``.
    """
    if k < 1 or k > iris_data.K**iris_data.N_FEATURES:
        raise ValueError(f"k must be in [1, 81], got {k}")
    rng = np.random.default_rng(seed)
    pats = valid_one_hot_patterns()
    draws = np.empty(n_draws, dtype=np.float64)
    for i in range(n_draws):
        idx = rng.choice(pats.shape[0], size=k, replace=False)
        draws[i] = covering_cost(pats[idx], codes)
    return float(draws.mean()), float(draws.std())


def covering_floor(codes: Codes, k: int) -> float:
    """Minimum ``R`` over ALL ``k``-subsets of the distinct codes — the best ``k`` medoids can
    do.  Brute force over ``itertools.combinations`` (``M ≤ ~25``, ``k = 4`` → ≤ 12 650
    subsets); guards ``k > M``.
    """
    uniq = np.unique(np.asarray(codes, dtype=np.int64))
    m = int(uniq.size)
    if k < 1 or k > m:
        raise ValueError(f"k > M: need 1 ≤ k ≤ {m}, got {k}")
    bits = iris_data.code_bits(uniq)
    return min(
        covering_cost(bits[list(subset)], uniq) for subset in itertools.combinations(range(m), k)
    )


@dataclass(frozen=True)
class PlacementScore:
    """One ``prototype_placement`` result; see that function for the semantics."""

    R: float
    """Mean halved Hamming distance from the distinct codes to the decoded patterns."""
    R_null: float
    """Mean ``R`` over random ``k``-subsets of the 81 valid patterns."""
    R_null_std: float
    """Standard deviation of the null distribution."""
    R_floor: float
    """``R`` of the best ``k``-subset of the codes themselves (brute-force optimum)."""
    rho: float
    """``(R_null − R)/(R_null − R_floor)``: 1 = "as good as the best ``k`` medoids",
    0 = "no better than random patterns" (negative = worse than random)."""
    n_dead: int
    """Hidden neurons with negligible output coupling (:func:`dead_hidden_neurons`)."""
    n_patterns: int
    """Patterns actually scored (``k`` after dropping dead neurons)."""
    encoding: str
    """The angle→probability map used to decode ("inverse" or "direct")."""
    all_dead: bool = False
    """Every neuron was flagged dead, so ``drop_dead`` was ignored and all ``k`` patterns
    scored — the score is then about the whole layer, dead or not."""


def prototype_placement(
    net: Network,
    params: npt.ArrayLike,
    codes: Codes,
    encoding: Encoding = "inverse",
    n_draws: int = 10_000,
    seed: int = 0,
    drop_dead: bool = True,
    tol_rel: float = 0.1,
    tol_abs: float = 0.0,
) -> PlacementScore:
    """Quantitative replacement for the ``[PTQNN Fig 9]`` t-SNE evidence.

    The hidden layer is decoded to one-hot patterns (:func:`decode_hidden_patterns`) and scored
    by :func:`covering_cost` against the distinct training codes, relative to the random-pattern
    null (:func:`covering_null`) and the brute-force medoid floor (:func:`covering_floor`).
    With ``drop_dead`` (default) neurons of negligible output coupling are excluded from the
    pattern set — ``k`` shrinks and null and floor use the reduced ``k`` — so a net that parked
    three of four neurons cannot inflate its coverage with dead ones.  If *every* neuron is
    flagged dead the function does NOT raise: it falls back to scoring all ``k`` patterns and
    reports ``all_dead=True`` with ``n_dead = k``.  (That case is real — with an
    amplitude-encoded target the trained couplings are uniformly small, see
    :func:`dead_hidden_neurons` — and losing a whole run to a diagnostic is worse than
    reporting a score that the ``all_dead`` flag qualifies.)

    Reproducing Fig 9's claim ("the prototype weights span the dataset", the classic ones do
    not) counts as demonstrated only if, over ≥ 20 (seed, split) pairs, the median
    ``rho_prototype ≥ 0.5`` with a bootstrap CI excluding 0, and ``rho_prototype > rho_classic``
    under a paired one-sided Wilcoxon test with ``p < 0.01`` ([session-4 analysis]).  This
    metric reads the ANGLES through an assumed angle→pattern map; see :func:`firing_placement`
    for the encoding-free companion that reads the neurons' firing instead.
    """
    patterns = decode_hidden_patterns(net, params, encoding=encoding)
    dead = dead_hidden_neurons(net, params, tol_rel=tol_rel, tol_abs=tol_abs)
    all_dead = bool(dead.all())
    kept = patterns[~dead] if (drop_dead and not all_dead) else patterns
    uniq = np.unique(np.asarray(codes, dtype=np.int64))
    r = covering_cost(kept, uniq)
    r_null, r_std = covering_null(uniq, kept.shape[0], n_draws=n_draws, seed=seed)
    r_floor = covering_floor(uniq, kept.shape[0])
    denom = r_null - r_floor
    # a vanishing denominator means random patterns are already optimal; anything real scores 1
    rho = 1.0 if denom < 1e-12 else (r_null - r) / denom
    return PlacementScore(
        R=r,
        R_null=r_null,
        R_null_std=r_std,
        R_floor=r_floor,
        rho=rho,
        n_dead=int(dead.sum()),
        n_patterns=int(kept.shape[0]),
        encoding=encoding,
        all_dead=all_dead,
    )


# --- encoding-free placement: what the hidden neurons actually FIRE on -------------------------
#
# The decoded-pattern metric above reads the ANGLES through an assumed angle→probability map,
# and the trained Iris prototypes defeat it ([session-5 analysis]).  On
# experiments/runs/iris/20260918T085641Z-drop the prototype neurons are ANTI-patterns: angles
# ≈ 0 on most inputs and RX(±π) on three or four of them — "a high weight before the CR-gate
# means that a 0-input is stressed" [Rosenhahn e-mail 2026-08-21] — so the per-block score
# vector is nearly constant and its argmax is a coin toss between three ties.  Whatever such a
# neuron is doing, `decode_hidden_patterns` cannot see it.
#
# The score below never decodes anything.  It asks the circuit: prepare a training code as a
# basis state on the input register, run the hidden layer, read P(hidden_j = 1).  That is the
# neuron's activation [PTQNN Fig 2, Fig 3] and it is defined for any angles, any encoding and
# either kind of neuron.


def _hidden_layer_factors(net: Network, params: npt.ArrayLike) -> tuple[Complex, list[Complex]]:
    """The hidden layer regrouped input qubit by input qubit — an exact rewriting.

    Returns ``(ρ₀, [U_0 … U_{n_inputs-1}])``: ``ρ₀`` is the hidden register after the part of
    the layer that touches no input qubit (the classic ``RX(bias_j)`` gates; the prototype
    neuron has no bias ``[PTQNN §III.B]``), and ``U_i`` is the ``(2, 2^n_hidden, 2, 2^n_hidden)``
    unitary — indexed ``[b_out, h_out, b_in, h_in]``, qubit 0 = MSB — of every gate that touches
    input qubit ``i``, on the ``1 + n_hidden`` qubit register "that input ⊗ hidden".

    Why this is exact.  Input qubits are never coupled to each other: the hidden layer is a
    product of blocks ``V_ij`` acting on ``(input i, hidden j)`` only.  Two blocks with
    ``i ≠ i'`` commute — either they are supported on disjoint qubits (``j ≠ j'``), or they
    share only ``h_j`` and there they are both ``X``-rotations by construction
    (``CRX`` ``[PTQNN Fig 1, Fig 2]``), which commute.  So the circuit may be reordered into
    per-input-qubit groups ``U_i = V_{i,n_hidden-1} ⋯ V_{i,0}`` (the ``j`` order inside a group
    is the circuit's own and does NOT commute).  Each input qubit is then touched by exactly one
    group and can be traced out immediately afterwards, which turns the layer into a sequence of
    ``n_inputs`` channels on the ``2^n_hidden``-dimensional hidden register:

        ρ ← Σ_{b∈{0,1}} K_b ρ K_b†,  K_b = ⟨b| U_i |c_i⟩  (a ``2^n_hidden`` square matrix).

    Cost ``O(n_inputs · 4^n_hidden)`` per code instead of ``O(gates · 2^(n_inputs+n_hidden))``
    — 13 ms instead of 2.5 s for the 12→4 Iris layer over 13 codes.  Pinned to the reference
    backend in ``test_firing_matrix_agrees_with_the_reference_backend``.
    """
    nh = net.n_hidden
    hidden, _ = net.unpack(params)
    dim = 2**nh
    rho0 = np.zeros((dim, dim), dtype=np.complex128)
    rho0[0, 0] = 1.0
    if net.hidden == "classic":
        bias = ir.Circuit(nh)
        for j, (_, b) in enumerate(hidden):
            bias.append(ir.rx(j, b))
        u_bias = batched.basis_columns(bias, range(nh))
        rho0 = u_bias @ rho0 @ u_bias.conj().T
    factors: list[Complex] = []
    for i in range(net.n_inputs):
        circ = ir.Circuit(1 + nh)
        for j, (angles, last) in enumerate(hidden):
            if net.hidden == "prototype":
                # [PTQNN Fig 2] RX(φ_i) · CRX(θ_j) · RX(−φ_i), the un-computation included
                circ.extend([ir.rx(0, angles[i]), ir.crx(0, 1 + j, last), ir.rx(0, -angles[i])])
            else:
                circ.append(ir.crx(0, 1 + j, angles[i]))  # [PTQNN Fig 1]
        factors.append(batched.basis_columns(circ, range(1 + nh)).reshape(2, dim, 2, dim))
    return rho0, factors


def _hidden_bit_table(n_hidden: int) -> Real:
    """``(2^n_hidden, n_hidden)`` — bit ``j`` of each hidden configuration (hidden qubit 0 = MSB,
    the convention of :func:`ptqnn.tasks.iris.model.output_activation_table`)."""
    idx = np.arange(2**n_hidden, dtype=np.int64)[:, None]
    return np.asarray((idx >> np.arange(n_hidden - 1, -1, -1)) & 1, dtype=np.float64)


def firing_matrix(
    net: Network, params: npt.ArrayLike, codes: Codes, backend: Backend | None = None
) -> Real:
    """``(n_hidden, M)`` — ``P(hidden_j = 1 | code c)`` for the ``M`` DISTINCT codes.

    Each code is prepared as a *basis* state on the input register (the per-sample read-out of
    :func:`ptqnn.tasks.iris.model.basis_activations`, not the superposition), the hidden layer
    is run and the marginal of hidden qubit ``j`` is read.  This is the activation the neuron
    figures of the paper plot ``[PTQNN Fig 3]``, measured on the data.

    ``backend=None`` uses the exact factorisation of :func:`_hidden_layer_factors`;
    passing a backend takes the oracle route — the FULL network circuit
    (:meth:`Network.circuit`, output neuron included) on the same basis state, with
    :func:`ptqnn.backends.reference.marginal` on each hidden qubit.  The two agree exactly: the
    output neuron only ever uses the hidden qubits as *controls* ``[PTQNN Fig 4]``, and a
    controlled gate leaves the reduced state of its control diagonal-blockwise untouched, so it
    cannot move ``P(hidden_j = 1)``.  For a prototype layer the value is the one *after* the
    ``RX(−φ)`` un-computation, i.e. the hidden qubit as the output neuron sees it.
    """
    uniq = np.unique(np.asarray(codes, dtype=np.int64))
    nh = net.n_hidden
    out = np.empty((nh, uniq.size), dtype=np.float64)
    if backend is not None:
        for m, code in enumerate(uniq):
            psi = iris_model.dataset_superposition(
                np.array([code], dtype=np.int64), net.n_qubits, net.input_qubits
            )
            p = backend.probabilities(net.circuit(params, initial_state=psi))
            for j, q in enumerate(net.hidden_qubits):
                out[j, m] = ref.marginal(p, [q])[1]
        return out
    rho0, factors = _hidden_layer_factors(net, params)
    bits = _hidden_bit_table(nh)
    for m, code in enumerate(uniq):
        rho = rho0
        for i, u in enumerate(factors):
            b_in = (int(code) >> (net.n_inputs - 1 - i)) & 1  # input qubit 0 = MSB of the code
            k0, k1 = u[0, :, b_in, :], u[1, :, b_in, :]
            rho = k0 @ rho @ k0.conj().T + k1 @ rho @ k1.conj().T
        out[:, m] = np.real(np.diag(rho)) @ bits
    return out


def coverage_selectivity(firing: npt.ArrayLike) -> tuple[float, float]:
    """``(coverage, selectivity)`` of an ``(n_hidden, M)`` firing matrix ``P_j(c)``.

    * ``coverage = mean_c max_j P_j(c)`` — *every data cluster has a neuron that responds to
      it*.  This is the firing-space reading of "the weights span the dataset"
      ``[PTQNN Fig 9]``; it is 1 when some neuron fires with certainty on every code and 0 when
      the layer ignores the data.
    * ``selectivity = mean_j (max_c P_j(c) − mean_c P_j(c))`` — *a neuron responds to some codes
      and not to others*.  A neuron with a constant response carries no information about the
      input however loudly it fires, and scores 0; coverage alone would reward it.

    ``[ASSUMPTION]`` both summaries.  The paper offers no quantitative placement criterion at
    all (Fig 9 is a t-SNE picture), so the reduction of "prototypes sit on the data" to these
    two numbers is ours.
    """
    f = np.asarray(firing, dtype=np.float64)
    if f.ndim != 2 or f.size == 0:
        raise ValueError(f"firing must be a non-empty (n_hidden, M) matrix, got {f.shape}")
    coverage = float(f.max(axis=0).mean())
    selectivity = float((f.max(axis=1) - f.mean(axis=1)).mean())
    return coverage, selectivity


@dataclass(frozen=True)
class FiringScore:
    """One :func:`firing_placement` result; see that function for the semantics."""

    coverage: float
    """``mean_c max_j P_j(c)`` of the trained layer."""
    selectivity: float
    """``mean_j (max_c P_j(c) − mean_c P_j(c))`` of the trained layer."""
    coverage_null: float
    """Mean coverage over ``n_draws`` random parameter vectors."""
    coverage_null_std: float
    """Standard deviation of the null coverage."""
    selectivity_null: float
    """Mean selectivity over the same draws."""
    selectivity_null_std: float
    """Standard deviation of the null selectivity."""
    rho_f: float
    """``(coverage − coverage_null)/coverage_null_std`` — a z-score against untrained layers."""
    rho_f_selectivity: float
    """The same z-score for the selectivity."""
    n_draws: int
    """Size of the random-parameter null."""
    n_codes: int
    """``M``, the number of distinct codes scored."""
    n_hidden: int
    """Neurons in the layer."""
    init_scale: float
    """Half-width of the uniform initialisation the null was drawn with."""


def firing_placement(
    net: Network,
    params: npt.ArrayLike,
    codes: Codes,
    n_draws: int = 200,
    seed: int = 0,
    init_scale: float = IrisConfig.init_scale,
    backend: Backend | None = None,
) -> FiringScore:
    """Encoding-free companion of :func:`prototype_placement`, scored on the neurons' FIRING.

    :func:`firing_matrix` gives ``P_j(c)`` on the distinct training codes;
    :func:`coverage_selectivity` reduces it to ``coverage`` (every code has a responder) and
    ``selectivity`` (a responder discriminates).  Both are compared with the layer the network
    *started* from: ``n_draws`` parameter vectors from ``net.random_params(rng, init_scale)``
    with ``rng = np.random.default_rng(seed)`` — the very initialisation
    :func:`train` uses (``IrisConfig.init_scale``, ``[ASSUMPTION]`` ±0.7 rad), and the same
    ``seed`` convention as :func:`covering_null` and the experiment script, which pass the run's
    seed.  The z-scores

        ``rho_f = (coverage − mean_null)/std_null``  and the same for the selectivity

    say by how many null standard deviations training moved the layer.  ``[ASSUMPTION]`` the
    z-score (rather than, say, a rank) and the choice of the *initialisation* distribution as
    the null: it is the only parameter distribution the paper fixes anything about, and it makes
    "the prototype layer ends up on the data" a statement about what training did, not about the
    architecture's prior.  A degenerate null (``std < 1e-12``) yields ``0.0``.

    Reproducing ``[PTQNN Fig 9]``'s claim in these terms means: over ≥ 20 (seed, split) pairs,
    the prototype network has ``rho_f > 0`` AND ``rho_f_selectivity > 0`` (training moved both
    summaries up, not just one), and both exceed the classic network's on the same seeds.  A
    single seed proves nothing — the null std is ~0.01–0.05 in coverage, so individual z-scores
    of a few units are routine.
    """
    if n_draws < 1:
        raise ValueError(f"n_draws must be ≥ 1, got {n_draws}")
    uniq = np.unique(np.asarray(codes, dtype=np.int64))
    coverage, selectivity = coverage_selectivity(firing_matrix(net, params, uniq, backend))
    rng = np.random.default_rng(seed)
    null = np.empty((n_draws, 2), dtype=np.float64)
    for d in range(n_draws):
        x = net.random_params(rng, scale=init_scale)
        null[d] = coverage_selectivity(firing_matrix(net, x, uniq, backend))
    mean, std = null.mean(axis=0), null.std(axis=0)

    def z(value: float, mu: float, sigma: float) -> float:
        return 0.0 if sigma < 1e-12 else (value - mu) / sigma

    return FiringScore(
        coverage=coverage,
        selectivity=selectivity,
        coverage_null=float(mean[0]),
        coverage_null_std=float(std[0]),
        selectivity_null=float(mean[1]),
        selectivity_null_std=float(std[1]),
        rho_f=z(coverage, float(mean[0]), float(std[0])),
        rho_f_selectivity=z(selectivity, float(mean[1]), float(std[1])),
        n_draws=int(n_draws),
        n_codes=int(uniq.size),
        n_hidden=int(net.n_hidden),
        init_scale=float(init_scale),
    )
