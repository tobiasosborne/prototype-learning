"""Reproduce ``[PTQNN Fig 7, Fig 8, Fig 9]`` — Iris with a classic and a prototype hidden layer.

    uv run python experiments/iris_fig7_fig8_fig9.py --seed 0 --iterations 40 --n-hidden 4

Writes ``experiments/runs/iris/<timestamp>/`` with ``meta.json`` (seed, git hash, config, split
statistics), ``losses.json``, ``params-<kind>.npy`` and the three figures.  The two models are
trained in parallel processes; each is a 17-qubit statevector optimisation and single-threaded,
so ``OMP_NUM_THREADS=1`` is forced.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ptqnn.circuits.neurons import HiddenKind
from ptqnn.tasks.iris import data as iris_data
from ptqnn.tasks.iris import model as iris_model
from ptqnn.tasks.iris import train as iris_train

KINDS: tuple[HiddenKind, HiddenKind] = ("classic", "prototype")
TITLE = {"classic": "classic perceptron model", "prototype": "prototype model"}


def git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        return "unknown"


def _train_one(
    args: tuple[HiddenKind, iris_train.IrisConfig, iris_data.IrisSplit, int],
) -> tuple[HiddenKind, iris_train.IrisResult]:
    kind, cfg, split, seed = args
    net = iris_train.network(cfg, kind)
    res = iris_train.train(
        net,
        split.train_codes,
        split.train_labels,
        split.test_codes,
        split.test_labels,
        cfg,
        seed=seed,
    )
    return kind, res


# [PTQNN Fig 7] read-offs (±0.005): left panel, classic, 15 epochs, train → 0.000, test dips to
# 0.030 at epoch 10 and rises to 0.045; right panel, prototype, 26 epochs, train → 0.010,
# test → 0.020.
PAPER_FIG7 = {
    "classic": {"train": 0.000, "test": 0.045, "epochs": 15},
    "prototype": {"train": 0.010, "test": 0.020, "epochs": 26},
}


def figure7(losses: dict[str, dict[str, list[float]]], path: Path) -> None:
    """Two panels, train (blue) and test (red) KL per sweep — the layout of ``[PTQNN Fig 7]``."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, kind in zip(axes, KINDS, strict=True):
        ax.plot(losses[kind]["train"], color="tab:blue", lw=2, label="Loss Train")
        ax.plot(losses[kind]["test"], color="tab:red", lw=2, label="Loss Test")
        ref = PAPER_FIG7[kind]
        ax.axhline(ref["train"], color="tab:blue", ls=":", lw=1, label="paper train (Fig 7)")
        ax.axhline(ref["test"], color="tab:red", ls=":", lw=1, label="paper test (Fig 7)")
        ax.axvline(ref["epochs"], color="0.6", ls="--", lw=1, label="paper epochs")
        ax.set_xlabel("KW–SA sweep")
        ax.set_ylabel("KL(target ‖ model)")
        ax.set_title(TITLE[kind])
        ax.set_ylim(bottom=0.0)
        ax.legend(fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _hist_panel(
    ax: plt.Axes, codes: np.ndarray, values: np.ndarray, title: str, ymax: float
) -> None:
    ax.bar(np.arange(codes.size), values, color="tab:blue")
    ax.set_ylim(0.0, ymax)
    ax.set_xticks(np.arange(codes.size))
    ax.set_xticklabels(
        [iris_data.format_code(int(c)) for c in codes], rotation=90, fontsize=5, family="monospace"
    )
    ax.set_title(title, fontsize=9)


def figure8_for(
    net_kind: HiddenKind,
    cfg: iris_train.IrisConfig,
    split: iris_data.IrisSplit,
    params: np.ndarray,
    path: Path,
) -> None:
    net = iris_train.network(cfg, net_kind)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for row, (name, codes, labels) in enumerate(
        (
            ("training", split.train_codes, split.train_labels),
            ("test", split.test_codes, split.test_labels),
        )
    ):
        uniq, values = iris_model.regressed_values(net, params, codes)
        target = iris_model.target_distribution(codes, labels)[uniq]
        ymax = 1.08 * float(max(values.max(), target.max()))
        _hist_panel(axes[row, 0], uniq, values, f"{name}: prediction ({TITLE[net_kind]})", ymax)
        _hist_panel(axes[row, 1], uniq, target, f"{name}: ground truth (amplitude ∝ label+1)", ymax)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figure9(
    cfg: iris_train.IrisConfig,
    split: iris_data.IrisSplit,
    results: dict[HiddenKind, iris_train.IrisResult],
    seed: int,
    path: Path,
) -> None:
    """``[PTQNN Fig 9]``: TSNE of the training codes (red) and the hidden weights (blue)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for ax, kind in zip(axes, KINDS, strict=True):
        net = iris_train.network(cfg, kind)
        w = iris_train.hidden_layer_weights_for_tsne(net, results[kind].params, as_pattern=True)
        data_xy, w_xy = iris_train.tsne_embedding(split.train_codes, w, seed=seed)
        ax.scatter(data_xy[:, 0], data_xy[:, 1], c="tab:red", s=40, label="training data")
        ax.scatter(w_xy[:, 0], w_xy[:, 1], c="tab:blue", s=90, marker="D", label="hidden weights")
        ax.set_title(f"TSNE — {TITLE[kind]}")
        ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def replot(out: Path) -> int:
    """Re-render the figures of a finished run from its saved ``losses.json`` and parameters."""
    meta = json.loads((out / "meta.json").read_text())
    cfg = iris_train.IrisConfig(**meta["config"])
    split = iris_data.load_split(seed=meta["seed"], train_size=float(meta["split"]["train_size"]))
    figure7(json.loads((out / "losses.json").read_text()), out / "fig07-iris-loss.png")
    params = {k: np.load(out / f"params-{k}.npy") for k in KINDS}
    for kind in KINDS:
        figure8_for(kind, cfg, split, params[kind], out / f"fig08-iris-hist-{kind}.png")
    results = {k: iris_train.IrisResult(params=params[k], hidden=k) for k in KINDS}
    figure9(cfg, split, results, meta["seed"], out / "fig09-iris-tsne.png")
    print(json.dumps({"replotted": str(out)}))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iterations", type=int, default=iris_train.IrisConfig.iterations)
    ap.add_argument("--n-hidden", type=int, default=iris_train.IrisConfig.n_hidden)
    ap.add_argument("--train-size", type=float, default=1.0 / 3.0)
    ap.add_argument("--init-scale", type=float, default=iris_train.IrisConfig.init_scale)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--replot",
        type=Path,
        default=None,
        help="re-render the figures of an existing run directory without retraining",
    )
    args = ap.parse_args(argv)

    if args.replot is not None:
        return replot(args.replot)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or Path("experiments/runs/iris") / stamp
    out.mkdir(parents=True, exist_ok=True)

    split = iris_data.load_split(seed=args.seed, train_size=args.train_size)
    cfg = iris_train.IrisConfig(
        iterations=args.iterations, n_hidden=args.n_hidden, init_scale=args.init_scale
    )

    from multiprocessing import Pool

    with Pool(2) as pool:
        trained = dict(pool.map(_train_one, [(k, cfg, split, args.seed) for k in KINDS]))
    results: dict[HiddenKind, iris_train.IrisResult] = trained

    accuracy: dict[str, dict[str, float]] = {}
    for kind in KINDS:
        net = iris_train.network(cfg, kind)
        p = results[kind].params
        np.save(out / f"params-{kind}.npy", p)
        uniq, act = iris_model.basis_activations(net, p, split.test_codes)
        per_sample = iris_model.labels_from_values(act)[np.searchsorted(uniq, split.test_codes)]
        psi = iris_model.dataset_superposition(split.train_codes, net.n_qubits, net.input_qubits)
        leak = 1.0 - float(
            iris_model.model_distribution(net, p, psi)[np.unique(split.train_codes)].sum()
        )
        accuracy[kind] = {
            "train": iris_model.accuracy(net, p, split.train_codes, split.train_labels),
            "test": iris_model.accuracy(net, p, split.test_codes, split.test_labels),
            "test_per_sample": float(np.mean(per_sample == split.test_labels)),
            "train_kl": results[kind].train_loss[-1],
            "test_kl": results[kind].test_loss[-1],
            "input_register_leak": leak,
        }
        figure8_for(kind, cfg, split, p, out / f"fig08-iris-hist-{kind}.png")

    figure7(
        {k: {"train": results[k].train_loss, "test": results[k].test_loss} for k in KINDS},
        out / "fig07-iris-loss.png",
    )
    figure9(cfg, split, results, args.seed, out / "fig09-iris-tsne.png")

    (out / "losses.json").write_text(
        json.dumps(
            {k: {"train": results[k].train_loss, "test": results[k].test_loss} for k in KINDS},
            indent=2,
        )
    )
    meta = {
        "experiment": "iris_fig7_fig8_fig9",
        "timestamp": stamp,
        "git_hash": git_hash(),
        "python": sys.version.split()[0],
        "seed": args.seed,
        "backend": "ptqnn.tasks.iris.model fast reduced statevector (exact)",
        "config": asdict(cfg),
        "split": split.summary(),
        "accuracy": accuracy,
        "paper": {
            "accuracy": "≈0.98 [PTQNN §III.C]",
            "fig7_classic": {"train_end": 0.000, "test_end": 0.045, "epochs": 15},
            "fig7_prototype": {"train_end": 0.010, "test_end": 0.020, "epochs": 26},
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"out": str(out), "accuracy": accuracy, "split": split.summary()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
