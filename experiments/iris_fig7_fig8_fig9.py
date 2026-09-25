"""Reproduce ``[PTQNN Fig 7, Fig 8, Fig 9]`` — Iris with a classic and a prototype hidden layer.

    uv run python experiments/iris_fig7_fig8_fig9.py --seed 0 --iterations 40 --n-hidden 4

Writes ``experiments/runs/iris/<UTC stamp>-seed<S>-<merge|drop>/`` with ``meta.json`` (seed,
git hash, host, platform, CPU count, worker processes, thread caps, wall time, config, split
statistics), ``losses.json``, ``results.json`` (accuracies and the two placement scores),
``params-<kind>.npy`` and the three figures.  The seed and the policy are in the directory name
so that a sweep may start many runs in the same second; a residual collision gets a ``-<pid>``
suffix (:func:`make_run_dir`).

CPU budget of one run: the two models train in ``--processes`` worker processes (default 2;
``--processes 1`` trains them one after the other in-process).  Each is a 17-qubit statevector
optimisation and single-threaded — ``OMP_NUM_THREADS``, ``OPENBLAS_NUM_THREADS`` and
``MKL_NUM_THREADS`` are set to 1 unless the caller's environment already sets them (that cap
also covers scikit-learn's k-means and t-SNE).  After training, scoring and plotting run in the
parent process alone.  So one run keeps at most ``--processes`` cores busy.

Order of writes: the trained parameters, ``losses.json`` and ``meta.json`` are on disk before
anything that merely *describes* the models runs (accuracies, placement scores, figures); each
placement scorer is wrapped as well.  A 40-minute training run must not be lost to a diagnostic
that raises (it was, once — ``prototype_placement`` refused an all-dead layer).

Accuracy under ``--ambiguous drop``: the drop policy removes test samples using their own labels
(review 004 M3), so ``results.json`` carries both the accuracy on the policy's test set
(``test``, ``test_per_sample``) and on every held-out sample (``test_unfiltered``,
``test_per_sample_unfiltered``); under ``merge`` the two coincide.

Two scoring-only modes need no training at all:

    uv run python experiments/iris_fig7_fig8_fig9.py --replot  RUN_DIR
    uv run python experiments/iris_fig7_fig8_fig9.py --rescore RUN_DIR

``--rescore`` rebuilds the split from the run's ``meta.json`` when there is one and refuses a
flag that contradicts it; without ``meta.json`` (a run that crashed before writing it) the flags
are used and ``meta-rescore.json`` records which value came from where.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "runs" / "iris"

KINDS: tuple[HiddenKind, HiddenKind] = ("classic", "prototype")
TITLE = {"classic": "classic perceptron model", "prototype": "prototype model"}


# Split/initialisation settings a run is defined by, with their defaults.  The argparse
# defaults are None so that --rescore can tell "given on the command line" from "defaulted".
SPLIT_DEFAULTS: dict[str, Any] = {
    "seed": 0,
    "train_size": 1.0 / 3.0,
    "ambiguous": "merge",
    "init_scale": iris_train.IrisConfig.init_scale,
}


def git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=ROOT
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        return "unknown"


def machine() -> dict[str, Any]:
    """Which machine and which code produced a run directory."""
    return {
        "git_hash": git_hash(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "threads": {v: os.environ.get(v) for v in THREAD_VARS},
    }


def make_run_dir(root: Path, seed: int, ambiguous: str, stamp: str | None = None) -> Path:
    """Create and return ``root/<stamp>-seed<seed>-<ambiguous>``, never one another run holds.

    ``mkdir`` without ``exist_ok`` is atomic, so two runs that collide on the name (same second,
    seed and policy) cannot both get it; the loser appends its pid.
    """
    stamp = stamp or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base = f"{stamp}-seed{seed}-{ambiguous}"
    root.mkdir(parents=True, exist_ok=True)
    for name in (base, f"{base}-{os.getpid()}", *(f"{base}-{os.getpid()}-{i}" for i in range(99))):
        try:
            (root / name).mkdir()
        except FileExistsError:
            continue
        return root / name
    raise FileExistsError(f"no free run directory for {base} under {root}")


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


def _per_sample_accuracy(net: Any, p: np.ndarray, codes: np.ndarray, labels: np.ndarray) -> float:
    """Accuracy of the per-sample read-out: every code prepared as a basis state."""
    uniq, act = iris_model.basis_activations(net, p, codes)
    per_sample = iris_model.labels_from_values(act)[np.searchsorted(uniq, codes)]
    return float(np.mean(per_sample == labels))


def evaluate_kind(
    cfg: iris_train.IrisConfig, split: iris_data.IrisSplit, kind: HiddenKind, p: np.ndarray
) -> dict[str, float]:
    """Train/test accuracy, the per-sample read-out and the input-register leak of one model.

    ``test``/``test_per_sample`` are on the policy's test set; ``*_unfiltered`` on every
    held-out sample (review 004 M3).  Under "drop" the unfiltered set contains codes that carry
    two labels, so no function of the code gets all of them right — that is the point: it is
    the ordinary held-out accuracy, the filtered one is conditional on the test labels.
    """
    net = iris_train.network(cfg, kind)
    psi = iris_model.dataset_superposition(split.train_codes, net.n_qubits, net.input_qubits)
    leak = 1.0 - float(
        iris_model.model_distribution(net, p, psi)[np.unique(split.train_codes)].sum()
    )
    return {
        "train": iris_model.accuracy(net, p, split.train_codes, split.train_labels),
        "test": iris_model.accuracy(net, p, split.test_codes, split.test_labels),
        "test_per_sample": _per_sample_accuracy(net, p, split.test_codes, split.test_labels),
        "test_unfiltered": iris_model.accuracy(net, p, split.test_codes_all, split.test_labels_all),
        "test_per_sample_unfiltered": _per_sample_accuracy(
            net, p, split.test_codes_all, split.test_labels_all
        ),
        "n_test": int(split.test_codes.size),
        "n_test_unfiltered": int(split.test_codes_all.size),
        "input_register_leak": leak,
    }


def print_accuracy(accuracy: dict[str, dict[str, float]], ambiguous: str) -> None:
    """One line per model: the policy's test accuracy AND the unfiltered one, side by side."""
    for kind in KINDS:
        a = accuracy[kind]
        print(
            f"accuracy  {kind:9s} train {a['train']:.4f}  "
            f"test ({ambiguous}) {a['test']:.4f} on {a['n_test']}  "
            f"test (unfiltered) {a['test_unfiltered']:.4f} on {a['n_test_unfiltered']}  "
            f"per-sample {a['test_per_sample']:.4f} / {a['test_per_sample_unfiltered']:.4f}"
        )


Scores = dict[str, dict[str, dict[str, object]]]


def score_placement(
    cfg: iris_train.IrisConfig,
    split: iris_data.IrisSplit,
    params: dict[HiddenKind, np.ndarray],
    seed: int,
    placement_draws: int,
    firing_draws: int,
) -> Scores:
    """``{"placement": ..., "firing": ...}`` — the two Fig-9 replacements, failures recorded.

    ``placement`` holds a :class:`~ptqnn.tasks.iris.train.PlacementScore` per
    (kind, encoding) — the decoded-pattern covering metric (pi-04); ``firing`` holds a
    :class:`~ptqnn.tasks.iris.train.FiringScore` per kind — the encoding-free score read off the
    neurons' actual activation on the training codes.  Every scorer is wrapped: an exception is
    stored as ``{"error": "..."}`` under its key rather than killing the run.
    """
    placement: dict[str, dict[str, object]] = {}
    firing: dict[str, dict[str, object]] = {}
    for kind in KINDS:
        net = iris_train.network(cfg, kind)
        for enc in ("inverse", "direct"):
            try:
                score = iris_train.prototype_placement(
                    net,
                    params[kind],
                    split.train_codes,
                    encoding=enc,
                    n_draws=placement_draws,
                    seed=seed,
                )
            except Exception as exc:  # a diagnostic must never lose a run
                placement[f"{kind}-{enc}"] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"placement {kind:9s} {enc:7s} FAILED: {type(exc).__name__}: {exc}")
                continue
            placement[f"{kind}-{enc}"] = asdict(score)
            print(
                f"placement {kind:9s} {enc:7s} R={score.R:.3f} null={score.R_null:.3f} "
                f"floor={score.R_floor:.3f} rho={score.rho:+.3f} dead={score.n_dead}"
                f"{' (ALL dead)' if score.all_dead else ''}"
            )
        try:
            fscore = iris_train.firing_placement(
                net,
                params[kind],
                split.train_codes,
                n_draws=firing_draws,
                seed=seed,
                init_scale=cfg.init_scale,
            )
        except Exception as exc:  # idem
            firing[kind] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"firing    {kind:9s} FAILED: {type(exc).__name__}: {exc}")
            continue
        firing[kind] = asdict(fscore)
        print(
            f"firing    {kind:9s} coverage={fscore.coverage:.3f} "
            f"(null {fscore.coverage_null:.3f}±{fscore.coverage_null_std:.3f}, "
            f"z={fscore.rho_f:+.2f})  selectivity={fscore.selectivity:.3f} "
            f"(null {fscore.selectivity_null:.3f}±{fscore.selectivity_null_std:.3f}, "
            f"z={fscore.rho_f_selectivity:+.2f})"
        )
    return {"placement": placement, "firing": firing}


def _recorded_settings(meta: dict[str, Any]) -> dict[str, Any]:
    """The split/initialisation settings a training ``meta.json`` recorded (absent keys omitted)."""
    found: dict[str, Any] = {}
    if "seed" in meta:
        found["seed"] = int(meta["seed"])
    split = meta.get("split", {})
    if "train_size" in split:
        found["train_size"] = float(split["train_size"])
    if "ambiguous" in split:
        found["ambiguous"] = str(split["ambiguous"])
    config = meta.get("config", {})
    if "init_scale" in config:
        found["init_scale"] = float(config["init_scale"])
    return found


def _agrees(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return bool(a == b)


def resolve_settings(
    args: argparse.Namespace, meta: dict[str, Any] | None
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    """``(values, sources, cli)`` for the keys of :data:`SPLIT_DEFAULTS`.

    A value recorded in ``meta`` wins ("meta.json"); otherwise the command line ("cli");
    otherwise the default ("default").  A command-line value that contradicts ``meta`` is an
    error: the run's own record beats the caller's memory (review 004 M7).
    """
    cli = {k: getattr(args, k) for k in SPLIT_DEFAULTS}
    recorded = _recorded_settings(meta) if meta is not None else {}
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    conflicts: list[str] = []
    for key, default in SPLIT_DEFAULTS.items():
        if key in recorded:
            if cli[key] is not None and not _agrees(cli[key], recorded[key]):
                flag = "--" + key.replace("_", "-")
                conflicts.append(
                    f"{flag} {cli[key]!r} contradicts meta.json {key}={recorded[key]!r}"
                )
            values[key], sources[key] = recorded[key], "meta.json"
        elif cli[key] is not None:
            values[key], sources[key] = cli[key], "cli"
        else:
            values[key], sources[key] = default, "default"
    if conflicts:
        raise SystemExit("--rescore refused: " + "; ".join(conflicts))
    return values, sources, cli


def rescore(out: Path, args: argparse.Namespace) -> int:
    """Re-evaluate a finished run from its ``params-<kind>.npy`` — no training.

    The split is rebuilt from ``seed``/``train_size``/``ambiguous`` and the firing null from
    ``init_scale``.  If the run directory holds a training ``meta.json`` these are read from it
    and a contradicting flag aborts before anything is written; otherwise the flags (or their
    defaults) are used.  ``meta-rescore.json`` records each value's source.  ``n_hidden`` is
    inferred from the parameter count, so it cannot be got wrong.  Writes
    ``results-rescore.json`` and ``meta-rescore.json`` into the run directory and touches
    nothing else.
    """
    t0 = time.perf_counter()
    meta_path = out / "meta.json"
    original = json.loads(meta_path.read_text()) if meta_path.exists() else None
    values, sources, cli = resolve_settings(args, original)
    params = {k: np.load(out / f"params-{k}.npy") for k in KINDS}
    sizes = {int(p.size) for p in params.values()}
    if len(sizes) != 1:
        raise ValueError(f"params-*.npy disagree on the parameter count: {sizes}")
    # n_params = n_hidden·(12 + 1) + n_hidden + 1
    size = sizes.pop()
    n_hidden, rest = divmod(size - 1, iris_data.N_BITS + 2)
    if rest or n_hidden < 1:
        raise ValueError(f"{size} parameters is not n_hidden·14 + 1 for a 12-input network")
    iterations = (original or {}).get("config", {}).get("iterations", args.iterations)
    cfg = iris_train.IrisConfig(
        iterations=int(iterations), n_hidden=n_hidden, init_scale=values["init_scale"]
    )
    split = iris_data.load_split(
        seed=values["seed"], train_size=values["train_size"], ambiguous=values["ambiguous"]
    )
    accuracy: dict[str, dict[str, float]] = {}
    for kind in KINDS:
        net = iris_train.network(cfg, kind)
        train_kl, test_kl = iris_model.train_test_losses(
            net,
            params[kind],
            split.train_codes,
            split.train_labels,
            split.test_codes,
            split.test_labels,
        )
        accuracy[kind] = evaluate_kind(cfg, split, kind, params[kind]) | {
            "train_kl": train_kl,
            "test_kl": test_kl,
        }
    scores = score_placement(
        cfg, split, params, values["seed"], args.placement_draws, args.firing_draws
    )
    (out / "results-rescore.json").write_text(
        json.dumps({"accuracy": accuracy, **scores}, indent=2)
    )
    (out / "meta-rescore.json").write_text(
        json.dumps(
            {
                "experiment": "iris_fig7_fig8_fig9 --rescore",
                "source_run": str(out),
                "source_meta": None
                if original is None
                else {"path": str(meta_path), "git_hash": original.get("git_hash")},
                "timestamp": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
                **machine(),
                "wall_seconds": time.perf_counter() - t0,
                "backend": "ptqnn.tasks.iris.model fast reduced statevector (exact)",
                "flags": values
                | {"placement_draws": args.placement_draws, "firing_draws": args.firing_draws},
                "sources": sources,
                "cli": cli,
                "n_hidden_from_params": n_hidden,
                "config": asdict(cfg),
                "split": split.summary(),
                "note": "flags: the settings used; sources: meta.json (the training run's own "
                "record), cli, or default; git_hash/hostname are those of THIS rescore",
            },
            indent=2,
        )
    )
    print(json.dumps({"rescored": str(out), "accuracy": accuracy}, indent=2))
    print_accuracy(accuracy, values["ambiguous"])
    return 0


def replot(out: Path) -> int:
    """Re-render the figures of a finished run from its saved ``losses.json`` and parameters."""
    meta = json.loads((out / "meta.json").read_text())
    cfg = iris_train.IrisConfig(**meta["config"])
    # runs predating the ambiguous flag have no policy recorded; the default was merge
    ambiguous = meta.get("split", {}).get("ambiguous", "merge")
    split = iris_data.load_split(
        seed=meta["seed"], train_size=float(meta["split"]["train_size"]), ambiguous=ambiguous
    )
    figure7(json.loads((out / "losses.json").read_text()), out / "fig07-iris-loss.png")
    params = {k: np.load(out / f"params-{k}.npy") for k in KINDS}
    for kind in KINDS:
        figure8_for(kind, cfg, split, params[kind], out / f"fig08-iris-hist-{kind}.png")
    results = {k: iris_train.IrisResult(params=params[k], hidden=k) for k in KINDS}
    figure9(cfg, split, results, meta["seed"], out / "fig09-iris-tsne.png")
    print(json.dumps({"replotted": str(out)}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=None, help="default 0")
    ap.add_argument("--iterations", type=int, default=iris_train.IrisConfig.iterations)
    ap.add_argument("--n-hidden", type=int, default=iris_train.IrisConfig.n_hidden)
    ap.add_argument("--train-size", type=float, default=None, help="default 1/3")
    ap.add_argument(
        "--ambiguous",
        choices=("merge", "drop"),
        default=None,
        help="policy for codes carrying two class labels [Rosenhahn e-mail 2026-08-21]; "
        '"drop" removes them entirely (default "merge"); part of the run directory name',
    )
    ap.add_argument(
        "--init-scale",
        type=float,
        default=None,
        help=f"default {iris_train.IrisConfig.init_scale}",
    )
    ap.add_argument(
        "--processes",
        type=int,
        choices=(1, 2),
        default=2,
        help="worker processes for the two (single-threaded) trainings; 1 trains them in turn "
        "in-process — use it when a sweep already parallelises over seeds",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--replot",
        type=Path,
        default=None,
        help="re-render the figures of an existing run directory without retraining",
    )
    ap.add_argument(
        "--rescore",
        type=Path,
        default=None,
        help="recompute accuracies and both placement scores of an existing run directory "
        "from its params-*.npy without retraining, writing results-rescore.json and "
        "meta-rescore.json; seed/split/init-scale are read from its meta.json if present",
    )
    ap.add_argument(
        "--placement-draws",
        type=int,
        default=10_000,
        help="random one-hot k-subsets in the covering-cost null (prototype_placement)",
    )
    ap.add_argument(
        "--firing-draws",
        type=int,
        default=1000,
        help="random parameter vectors in the firing-score null (firing_placement); ≥ 1000 "
        "for any statistical statement",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.replot is not None:
        return replot(args.replot)
    if args.rescore is not None:
        return rescore(args.rescore, args)

    t0 = time.perf_counter()
    settings = {
        k: (v if (v := getattr(args, k)) is not None else d) for k, d in SPLIT_DEFAULTS.items()
    }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.out is not None:
        out = args.out
        out.mkdir(parents=True, exist_ok=True)
    else:
        out = make_run_dir(RUNS, settings["seed"], settings["ambiguous"], stamp)

    split = iris_data.load_split(
        seed=settings["seed"], train_size=settings["train_size"], ambiguous=settings["ambiguous"]
    )
    cfg = iris_train.IrisConfig(
        iterations=args.iterations, n_hidden=args.n_hidden, init_scale=settings["init_scale"]
    )

    jobs = [(k, cfg, split, settings["seed"]) for k in KINDS]
    if args.processes == 1:
        trained = dict(_train_one(job) for job in jobs)
    else:
        from multiprocessing import Pool

        with Pool(args.processes) as pool:
            trained = dict(pool.map(_train_one, jobs))
    results: dict[HiddenKind, iris_train.IrisResult] = trained
    wall_training = time.perf_counter() - t0

    # Everything that costs 40 minutes is on disk before anything that merely describes it runs:
    # no accuracy, scorer or plot can lose the run any more.
    params = {k: results[k].params for k in KINDS}
    for kind in KINDS:
        np.save(out / f"params-{kind}.npy", params[kind])
    losses = {k: {"train": results[k].train_loss, "test": results[k].test_loss} for k in KINDS}
    (out / "losses.json").write_text(json.dumps(losses, indent=2))
    meta: dict[str, Any] = {
        "experiment": "iris_fig7_fig8_fig9",
        "timestamp": stamp,
        **machine(),
        "processes": args.processes,
        "wall_seconds_training": wall_training,
        "seed": settings["seed"],
        "backend": "ptqnn.tasks.iris.model fast reduced statevector (exact)",
        "config": asdict(cfg),
        "split": split.summary(),
        "evaluations": {k: results[k].evaluations for k in KINDS},
        "paper": {
            "accuracy": "≈0.98 [PTQNN §III.C]",
            "fig7_classic": {"train_end": 0.000, "test_end": 0.045, "epochs": 15},
            "fig7_prototype": {"train_end": 0.010, "test_end": 0.020, "epochs": 26},
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    accuracy: dict[str, dict[str, float]] = {}
    for kind in KINDS:
        accuracy[kind] = evaluate_kind(cfg, split, kind, params[kind]) | {
            "train_kl": results[kind].train_loss[-1],
            "test_kl": results[kind].test_loss[-1],
        }
    (out / "results.json").write_text(
        json.dumps({"accuracy": accuracy, "placement": {}, "firing": {}}, indent=2)
    )
    meta["accuracy"] = accuracy
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    # The two quantitative replacements for the Fig 9 t-SNE ([session-4/5 analysis]): the
    # decoded-pattern covering metric (pi-04, both encodings per kind) and the encoding-free
    # firing score.  Both are scored on the training codes; neither may abort the run.
    scores = score_placement(
        cfg, split, params, settings["seed"], args.placement_draws, args.firing_draws
    )
    (out / "results.json").write_text(json.dumps({"accuracy": accuracy, **scores}, indent=2))

    # figures last: they are the only step whose failure costs nothing but a re-``--replot``
    for kind in KINDS:
        figure8_for(kind, cfg, split, params[kind], out / f"fig08-iris-hist-{kind}.png")
    figure7(losses, out / "fig07-iris-loss.png")
    figure9(cfg, split, results, settings["seed"], out / "fig09-iris-tsne.png")
    meta["wall_seconds_total"] = time.perf_counter() - t0
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"out": str(out), "accuracy": accuracy, "split": split.summary()}, indent=2))
    print_accuracy(accuracy, settings["ambiguous"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
