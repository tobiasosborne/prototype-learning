"""Training and Table I for the ECG denoiser ``[PTQNN §III.D, Table I]``.

Table I compares "CQNN ... the classical quantum neural network with 1-3 neurons as latent space
dimension in the hidden layer" against "PTQNN ... the prototype model with 1-3 prototype layers",
by PSNR and SSIM.  :func:`table1` runs that 2 × 3 grid; :func:`train` is one cell of it.

Nothing about the optimisation of this experiment is published — no split, no iteration count, no
learning rate — so the defaults below are ours and marked ``[ASSUMPTION]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from ptqnn.backends.base import Backend
from ptqnn.circuits.neurons import HiddenKind, Network
from ptqnn.tasks.ecg import data, model
from ptqnn.train import kwsa

Real = npt.NDArray[np.float64]

# ``[PTQNN Table I]``, verbatim: rows are the number of latent-space neurons / prototype layers.
PAPER_TABLE1: dict[tuple[HiddenKind, int], tuple[float, float]] = {
    ("classic", 1): (32.72, 0.90),
    ("classic", 2): (38.20, 0.90),
    ("classic", 3): (39.65, 0.92),
    ("prototype", 1): (32.9, 0.87),
    ("prototype", 2): (40.00, 0.93),
    ("prototype", 3): (40.04, 0.93),
}

# [ASSUMPTION] KW–SA settings for the ECG regression.  The bit-count settings
# (``bitcount.SUPERPOSITION_CONFIG``) are tuned for a KL loss of order 0.05; here the loss is an
# MSE of order 1e-3, so the gradient — and with it the useful step size — is orders of magnitude
# smaller.  Grid-searched on synthetic data (100 sweeps, 1 and 2 hidden neurons, both kinds,
# a0 ∈ {20, 200, 1000, 2000}, c0 ∈ {0.05, 0.1, 0.2, 0.4, 1.0}, γ ∈ {0.101, 0.5, 0.9}):
#   * γ is what matters.  In coordinate mode a sweep keeps the best of {x ± c_n, x − a_n ĝ}, so
#     when the KW step overshoots — which it does early — the *finite-difference* step c_n is
#     the actual move.  c_n must therefore start coarse and end fine: γ = 0.9 takes c from 0.4
#     to 0.006 over 100 sweeps and beat γ = 0.101 by 1.5–1.9 dB on every configuration tried.
#   * a0 = 200 with α = 0.3; 1000 and 2000 were worse (overshoot), 20 no better.
#   * greedy acceptance (t0 = 0): annealing noise cost more than it bought on this loss.
DEFAULT_CONFIG = kwsa.KWSAConfig(
    iterations=100, a0=200.0, alpha=0.3, c0=0.4, gamma=0.9, t0=0.0, seed=0
)

# [ASSUMPTION] Windows per loss evaluation.  A coordinate sweep costs 3 evaluations per
# parameter (22 parameters for 3 prototypes), so a full-signal loss would be ~35 s per sweep.
# A fixed random subset of 600 of the ~1800 windows keeps the estimate unbiased and the sweep
# near 12 s; the reported metrics always use every window.
DEFAULT_SUBSAMPLE = 600

# [ASSUMPTION] Number of starting points screened by loss before the sweep begins; see
# :func:`init_candidates_params`.  256 costs a couple of seconds.  It narrows but does not close
# the seed-to-seed spread — a prototype cell can still land anywhere between 25 and 37 dB — which
# is why ``table1`` takes the best of several restarts.
DEFAULT_INIT_CANDIDATES = 256

# [ASSUMPTION] Initialisation.  ``init_candidates`` parameter vectors are screened by loss and
# the sweep starts at the best of them.  ``"random"`` draws uniform angles in [−π, π].
# ``"window"`` is the draw the paper's own reading of the prototype layer suggests — "the
# weights of the prototype layer [span] much better the dataset. These weights can be easier
# assigned to samples from the dataset" ``[PTQNN Fig 9]`` — namely ``φ_ji = −π·x_r`` for a random
# training window ``x_r``, which makes input qubit ``i`` carry the angle ``π(s_i − x_ri)`` inside
# the prototype block, i.e. a distance-to-prototype activation ``sin²(π(s_i − x_ri)/2)``.
# Measured (1 prototype, 60 sweeps, synthetic): window init *loses* — MSE 6.2e-3 against 8.7e-4
# for the random draw.  It starts every neuron in its dead zone: for a regression whose windows
# are all near the prototype, ``s_i − x_ri ≈ 0`` gives ``P(hidden = 1) ≈ 0`` and no gradient.
# The option is kept because that is a statement about *this* task, not about prototype layers.
InitKind = Literal["auto", "random", "window"]


def init_candidates_params(
    net: Network, rng: np.random.Generator, n: int, X: Real, init: InitKind = "auto"
) -> list[Real]:
    """``n`` starting points for KW–SA; ``"auto"`` is ``"random"`` (see the note above)."""
    kind = "random" if init == "auto" else init
    if kind == "random":
        return [net.random_params(rng) for _ in range(n)]
    out: list[Real] = []
    for _ in range(n):
        hidden = [
            (-np.pi * X[rng.integers(X.shape[0])], float(rng.uniform(-np.pi, np.pi)))
            for _ in range(net.n_hidden)
        ]
        # .tolist(): Network.pack types its `output` weights as Sequence[float] while `hidden`
        # accepts an array too — see the report's list of requested upstream changes.
        w_out = rng.uniform(-np.pi, np.pi, net.n_hidden).tolist()
        out.append(net.pack(hidden, (w_out, float(rng.uniform(-np.pi, np.pi)))))
    return out


Source = Literal["synthetic", "physionet", "mailed"]


@dataclass(frozen=True)
class EcgDataset:
    """One noisy ECG cut into training and test windows, with the clean signal kept for the
    metrics and for the Fig-13 plot."""

    clean: Real
    noisy: Real
    X: Real
    y: Real
    train_idx: npt.NDArray[np.int64]
    test_idx: npt.NDArray[np.int64]
    # Sigma the *noise* carries: the requested level for synthetic/physionet, the measured
    # ``std(noisy − clean)`` for ``source="mailed"`` (whose files arrive pre-noised).
    sigma: float
    source: Source

    @property
    def R(self) -> float:  # noqa: N802 — "R" is the paper's symbol (Eq 6)
        """ "The maximum fluctuation in the signal" ``[PTQNN §III.D]`` — see :func:`model.psnr`."""
        return float(np.ptp(self.clean))


def prepare(
    source: Source = "synthetic",
    n_samples: int | None = 1800,
    seed: int = 0,
    sigma: float = data.DEFAULT_SIGMA,
    train_frac: float = 0.5,
    half_width: int = 2,
    **source_kwargs: Any,
) -> EcgDataset:
    """Build the dataset: clean signal, additive noise, sliding windows, contiguous split.

    ``source="mailed"`` is the co-author's original recording: its files already carry the
    noise, so :func:`data.add_noise` is *not* applied and ``sigma`` comes back **measured** as
    ``std(noisy − clean)`` rather than as the requested ``sigma`` (which is ignored for this
    source).  ``n_samples=None`` keeps the whole trace, which only ``"mailed"`` supports.
    """
    if source == "synthetic":
        if n_samples is None:
            raise ValueError('source="synthetic" needs an explicit n_samples')
        clean = data.synthetic_ecg(n_samples, seed=seed, **source_kwargs)
        noisy = data.add_noise(clean, sigma=sigma, seed=seed + 1)
    elif source == "physionet":
        if n_samples is None:
            raise ValueError('source="physionet" needs an explicit n_samples')
        clean = data.load_physionet(n_samples=n_samples, **source_kwargs)
        noisy = data.add_noise(clean, sigma=sigma, seed=seed + 1)
    else:
        clean, noisy = data.load_mailed(n_samples=n_samples, **source_kwargs)
        sigma = float(np.std(noisy - clean))
    X, y = data.supervised_windows(noisy, clean, half_width=half_width)
    tr, te = data.contiguous_split(X.shape[0], train_frac=train_frac, half_width=half_width)
    return EcgDataset(clean, noisy, X, y, tr, te, sigma, source)


def train(
    net: Network,
    X: Real,
    y: Real,
    cfg: kwsa.KWSAConfig | None = None,
    seed: int = 0,
    iterations: int | None = None,
    subsample: int | None = DEFAULT_SUBSAMPLE,
    backend: Backend | None = None,
    init_candidates: int = DEFAULT_INIT_CANDIDATES,
    init: InitKind = "auto",
    encoding: model.Encoding = model.DEFAULT_ENCODING,
) -> kwsa.KWSAResult:
    """KW–SA on ``mean((P(out=1) − s(t))²)`` over the training windows ``[PTQNN §II.C]``."""
    config = cfg or DEFAULT_CONFIG
    overrides: dict[str, Any] = {"seed": seed}
    if iterations is not None:
        overrides["iterations"] = iterations
    config = kwsa.KWSAConfig(**{**config.__dict__, **overrides})
    loss = model.mse_loss(net, X, y, backend, subsample=subsample, seed=seed, encoding=encoding)
    rng = np.random.default_rng(seed)
    candidates = init_candidates_params(net, rng, max(init_candidates, 1), X, init)
    x0 = min(candidates, key=loss)
    return kwsa.minimise(loss, x0, config)


def evaluate(
    net: Network,
    params: Real,
    X: Real,
    y: Real,
    clean_segment: Real | None = None,
    backend: Backend | None = None,
    encoding: model.Encoding = model.DEFAULT_ENCODING,
) -> dict[str, float]:
    """PSNR/SSIM/MSE of the denoised windows against the clean centre samples.

    ``clean_segment`` supplies ``R`` (the maximum fluctuation of the *whole* clean signal); with
    it left out, ``R`` comes from ``y`` itself, which for a short excerpt understates the range.
    """
    yhat = model.predict(net, params, X, backend, encoding)
    R = float(np.ptp(clean_segment)) if clean_segment is not None else float(np.ptp(y))
    return {
        "mse": model.mse(y, yhat),
        "psnr": model.psnr(y, yhat, R),
        "ssim": model.ssim(y, yhat),
        "ssim_windowed": model.ssim_windowed(y, yhat),
    }


def run_cell(
    dataset: EcgDataset,
    n_hidden: int,
    kind: HiddenKind,
    seed: int = 0,
    iterations: int | None = None,
    subsample: int | None = DEFAULT_SUBSAMPLE,
    cfg: kwsa.KWSAConfig | None = None,
    init: InitKind = "auto",
    encoding: model.Encoding = model.DEFAULT_ENCODING,
) -> dict[str, Any]:
    """One cell of Table I: train on the training block, report on train, test and full signal."""
    net = Network(n_inputs=dataset.X.shape[1], n_hidden=n_hidden, hidden=kind)
    tr, te = dataset.train_idx, dataset.test_idx
    res = train(
        net,
        dataset.X[tr],
        dataset.y[tr],
        cfg=cfg,
        seed=seed,
        iterations=iterations,
        subsample=subsample,
        init=init,
        encoding=encoding,
    )
    paper = PAPER_TABLE1[(kind, n_hidden)]
    row: dict[str, Any] = {
        "kind": kind,
        "model": "PTQNN" if kind == "prototype" else "CQNN",
        "n_hidden": n_hidden,
        "n_params": net.n_params,
        "n_qubits": net.n_qubits,
        "encoding": encoding,
        "loss": res.loss,
        "evaluations": res.evaluations,
        "paper_psnr": paper[0],
        "paper_ssim": paper[1],
        "params": res.params.tolist(),
    }
    for name, idx in (("train", tr), ("test", te), ("full", np.arange(dataset.X.shape[0]))):
        out = evaluate(
            net, res.params, dataset.X[idx], dataset.y[idx], dataset.clean, None, encoding
        )
        row |= {f"{k}_{name}": v for k, v in out.items()}
    # The alternative reading of "maximum fluctuation" (R = 1, the full dynamic range of a
    # signal in [0, 1]) adds a constant 20 log10(1/R) dB; reported so that Table I can be
    # compared under either convention.  See :func:`model.psnr`.
    yhat_te = model.predict(net, res.params, dataset.X[te], None, encoding)
    row["psnr_test_R1"] = model.psnr(dataset.y[te], yhat_te, 1.0)
    # The noisy input itself, for reference: what "no denoising at all" scores.
    centre = dataset.X[:, dataset.X.shape[1] // 2]
    row["psnr_input"] = model.psnr(dataset.y, centre, dataset.R)
    row["ssim_input"] = model.ssim(dataset.y, centre)
    # Table I reports a single PSNR/SSIM pair; we quote the held-out block.
    row |= {"psnr": row["psnr_test"], "ssim": row["ssim_test"], "mse": row["mse_test"]}
    return row


def best_of(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pick the restart to report.  Model selection uses the *training* MSE — the test block is
    never consulted, so the reported PSNR/SSIM stay honest held-out numbers."""
    best = min(rows, key=lambda r: r["mse_train"])
    return {**best, "restarts": len(rows), "restart_mse_train": [r["mse_train"] for r in rows]}


def table1(
    n_hidden: Sequence[int] = (1, 2, 3),
    kinds: Sequence[HiddenKind] = ("classic", "prototype"),
    dataset: EcgDataset | None = None,
    seed: int = 0,
    iterations: int | None = None,
    subsample: int | None = DEFAULT_SUBSAMPLE,
    restarts: int = 1,
    encoding: model.Encoding = model.DEFAULT_ENCODING,
) -> list[dict[str, Any]]:
    """The full Table I grid, serially.  ``experiments/ecg_table1.py`` runs the same cells (and
    their restarts) in parallel; this function is the reference, and what the slow test
    exercises.

    [ASSUMPTION] ``restarts`` independent KW–SA runs per cell, keeping the one with the lowest
    training MSE.  The paper reports one number per cell and says nothing about restarts; a
    single greedy sweep on this landscape has a visible seed-to-seed spread, and the cells are
    embarrassingly parallel.
    """
    ds = dataset or prepare(seed=seed)
    return [
        best_of(
            [
                run_cell(
                    ds,
                    k,
                    kind,
                    seed=seed + r,
                    iterations=iterations,
                    subsample=subsample,
                    encoding=encoding,
                )
                for r in range(max(restarts, 1))
            ]
        )
        for kind in kinds
        for k in n_hidden
    ]
