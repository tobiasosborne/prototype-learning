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

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
from sklearn.manifold import TSNE

from ptqnn.backends.base import Backend
from ptqnn.circuits.neurons import HiddenKind, Network
from ptqnn.tasks.iris import model as iris_model
from ptqnn.tasks.iris.data import Codes, code_bits
from ptqnn.train import kwsa

Real = npt.NDArray[np.float64]

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
