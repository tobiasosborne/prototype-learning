"""Reproduce ``[PTQNN Table I]`` and ``[PTQNN Fig 13]``: sliding-window ECG denoising.

Runs the 2 × 3 grid (CQNN/PTQNN × 1/2/3 hidden neurons) — one process per cell, since KW–SA is
strictly sequential inside a cell — and writes ``params.npz``, ``meta.json``, ``results.json``
and a Fig-13-style ``fig13.png`` (noisy red, ground truth green, denoised blue) to
``experiments/runs/ecg/<timestamp>/``.

    uv run python experiments/ecg_table1.py --source synthetic --iterations 40
    uv run python experiments/ecg_table1.py --source physionet --record 100
    uv run python experiments/ecg_table1.py --source mailed --n-samples 1800
    uv run python experiments/ecg_table1.py --source mailed --n-samples 0  # all 7700 samples
"""

from __future__ import annotations

import os

# Before numpy: pool workers inherit the BLAS pool size fixed at import (one thread per worker).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import multiprocessing as mp
import platform
import socket
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ptqnn.circuits.neurons import HiddenKind
from ptqnn.tasks.ecg import data, model, train

ROOT = Path(__file__).resolve().parents[1]


def git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip()
    except Exception:  # provenance is best effort, never fatal
        return "unknown"


def provenance(processes: int) -> dict[str, Any]:
    """Which machine produced the run and with how many worker processes (review 004 B3)."""
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "processes": processes,
    }


def _cell(
    args: tuple[train.EcgDataset, int, HiddenKind, int, int, int, model.Encoding],
) -> dict[str, Any]:
    ds, k, kind, seed, iterations, subsample, encoding = args
    # elapsed times from the monotonic clock: the wall clock stepped backwards under WSL2 and the
    # archived run recorded wall_seconds = -6179 (review 004 B3)
    t0 = time.perf_counter()
    row = train.run_cell(
        ds, k, kind, seed=seed, iterations=iterations, subsample=subsample, encoding=encoding
    )
    row["seconds"] = time.perf_counter() - t0
    row["seed"] = seed
    print(
        f"  {row['model']:6s} {k} seed {seed}  train MSE {row['mse_train']:.3e}  "
        f"PSNR {row['psnr']:6.2f}  {row['seconds']:.0f}s",
        flush=True,
    )
    return row


def figure13(ds: train.EcgDataset, rows: list[dict[str, Any]], path: Path) -> None:
    """``[PTQNN Fig 13]``: "The red curve is the noisy input signal, the ground truth signal is
    shown in green and the denoising result of our prototype quantum neural network is shown in
    blue."  We plot the best prototype model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ptqnn.circuits.neurons import Network

    best = max((r for r in rows if r["kind"] == "prototype"), key=lambda r: r["psnr"])
    net = Network(n_inputs=5, n_hidden=best["n_hidden"], hidden="prototype")
    yhat = model.predict(net, np.asarray(best["params"]), ds.X, None, best["encoding"])

    fig, ax = plt.subplots(figsize=(16, 4))
    t = np.arange(ds.clean.size)
    ax.plot(t, ds.noisy, color="red", lw=1.2, label="Noisy Input")
    ax.plot(t, ds.clean, color="lime", lw=1.2, label="Ground truth")
    ax.plot(t, yhat, color="blue", lw=1.0, label="PT-QNN-Denoise")
    ax.axvline(ds.test_idx[0], color="0.6", ls="--", lw=0.8)
    ax.text(ds.test_idx[0] + 10, 0.95, "test", color="0.4", va="top", fontsize=9)
    ax.set_xlabel("t")
    ax.set_ylabel("amplitude")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, ds.clean.size)
    ax.legend(loc="upper right")
    ax.set_title(
        f"{ds.source} ECG, σ={ds.sigma}, PTQNN with {best['n_hidden']} prototype(s): "
        f"PSNR {best['psnr']:.2f} dB"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=("synthetic", "physionet", "mailed"), default="synthetic")
    p.add_argument("--record", default="100", help="PhysioNet record (--source physionet)")
    p.add_argument(
        "--mailed-dir",
        type=Path,
        default=data.MAILED_DIR,
        help="directory with the mailed ecgfilteredData.mat / ecgnoiseData.mat (--source mailed)",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=1800,
        help="signal length (Fig 13: 1800); 0 = the full trace (--source mailed: 7700)",
    )
    p.add_argument("--sigma", type=float, default=data.DEFAULT_SIGMA)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iterations", type=int, default=40)
    p.add_argument("--subsample", type=int, default=train.DEFAULT_SUBSAMPLE)
    p.add_argument("--train-frac", type=float, default=0.5)
    p.add_argument(
        "--encoding",
        choices=("amplitude", "linear"),
        default=model.DEFAULT_ENCODING,
        help="value->angle map: 2*asin(sqrt(s)) [SNN Fig 4] or pi*s",
    )
    p.add_argument("--restarts", type=int, default=8, help="KW-SA restarts per cell")
    p.add_argument("--processes", type=int, default=48)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    out = args.out or ROOT / "experiments" / "runs" / "ecg" / time.strftime("%Y%m%dT%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    kwargs: dict[str, Any] = {}
    if args.source == "physionet":
        kwargs["record"] = args.record
    elif args.source == "mailed":
        kwargs["directory"] = args.mailed_dir
    # 0 windows is meaningless in this task, so `--n-samples 0` is the sentinel for "the whole
    # trace"; only the mailed source has a variable length (the file's 7700 samples).
    n_samples: int | None = args.n_samples if args.n_samples > 0 else None
    ds = train.prepare(
        source=args.source,
        n_samples=n_samples,
        seed=args.seed,
        sigma=args.sigma,
        train_frac=args.train_frac,
        **kwargs,
    )
    print(
        f"{args.source}: {ds.clean.size} samples, R={ds.R:.3f}, σ={ds.sigma}, "
        f"{ds.train_idx.size} train / {ds.test_idx.size} test windows; "
        f"noisy input PSNR {model.psnr(ds.y, ds.X[:, 2], ds.R):.2f} dB → {out}",
        flush=True,
    )

    # 6 cells x `restarts` independent KW-SA runs, all in parallel; the best restart per cell
    # (by *training* MSE) is the reported row.  KW-SA itself is strictly sequential, so this is
    # the only parallelism available, and 6 x 8 = 48 jobs is what the 64 cores are for.
    jobs = [
        (ds, k, kind, args.seed + r, args.iterations, args.subsample, args.encoding)
        for kind in ("classic", "prototype")
        for k in (1, 2, 3)
        for r in range(args.restarts)
    ]
    processes = min(args.processes, len(jobs))
    t0 = time.perf_counter()
    with mp.Pool(processes=processes) as pool:
        every = pool.map(_cell, jobs)
    wall = time.perf_counter() - t0
    rows = [
        train.best_of([r for r in every if r["kind"] == kind and r["n_hidden"] == k])
        for kind in ("classic", "prototype")
        for k in (1, 2, 3)
    ]

    np.savez(
        out / "params.npz",
        **{f"{r['kind']}_{r['n_hidden']}": np.asarray(r["params"]) for r in rows},
        clean=ds.clean,
        noisy=ds.noisy,
    )
    meta = {
        "seed": args.seed,
        "git_hash": git_hash(),
        "source": args.source,
        "sigma": ds.sigma,
        "R": ds.R,
        "n_samples": ds.clean.size,
        "record": args.record if args.source == "physionet" else None,
        "mailed": (data.mailed_provenance(args.mailed_dir) if args.source == "mailed" else None),
        "config": {**asdict(train.DEFAULT_CONFIG), "iterations": args.iterations},
        "subsample": args.subsample,
        "restarts": args.restarts,
        "train_frac": args.train_frac,
        "encoding": args.encoding,
        "backend": "numpy statevector (batched)",
        **provenance(processes),
        "processes_requested": args.processes,
        "wall_seconds": wall,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (out / "results.json").write_text(
        json.dumps(
            {
                "input_psnr": model.psnr(ds.y, ds.X[:, 2], ds.R),
                "input_ssim": model.ssim(ds.y, ds.X[:, 2]),
                "input_ssim_windowed": model.ssim_windowed(ds.y, ds.X[:, 2]),
                "rows": rows,
                "all_restarts": every,
            },
            indent=2,
        )
        + "\n"
    )
    figure13(ds, rows, out / "fig13.png")

    print(f"\n{'':6s} {'#':>2}  {'PSNR':>7} {'paper':>6}  {'SSIM':>6} {'paper':>6}  {'SSIM_w':>6}")
    for r in rows:
        print(
            f"{r['model']:6s} {r['n_hidden']:2d}  {r['psnr']:7.2f} {r['paper_psnr']:6.2f}  "
            f"{r['ssim']:6.4f} {r['paper_ssim']:6.2f}  {r['ssim_windowed_test']:6.4f}"
        )
    print(f"\nwall {wall:.0f}s → {out}")


if __name__ == "__main__":
    mp.set_start_method("fork")
    main()
