"""Bit-count training on a *sampling* backend: does Kiefer–Wolfowitz survive shot noise, and what
do common random numbers buy?  (Shot-noise programme, item 2.)

Everything the paper trains is trained on exact probabilities.  On hardware the loss is computed
from ``counts/N``, so the central differences of ``[PTQNN §II.C]`` see shot noise of variance
``O(1/N)`` on *each* probe and ``O(1/(c²N))`` on the derivative.  This script trains the Fig-5
network (``Network(5, 5)``, 11 qubits, 36 parameters) with our reproduction's KW–SA schedule
(``bitcount.SUPERPOSITION_CONFIG`` — the paper publishes no schedule; ours is ``[ASSUMPTION]``)
on ``ShotBackend`` at ``N`` shots per evaluation, with and
without common random numbers (``kwsa.minimise(probe_context=backend.common_random_numbers)``),
and records the *exact* KL of the iterate after every sweep (monitored on a statevector, never
used for optimisation).

Grid: ``N ∈ {1000, 4000, 16000}`` × ``CRN ∈ {off, on}`` × 3 optimiser seeds, plus the exact
(``N = ∞``) reference for the same seeds.  One loss evaluation is ≈2.5 ms (statevector) + the
multinomial draw; 2 160 evaluations per run → the whole grid is a few minutes.  The script
measures the first run and **aborts** if the projection exceeds ``--budget-s``.

    uv run python experiments/bitcount_shots.py

writes ``experiments/runs/bitcount_shots/<stamp>/{results.json,results.md,figA_kl_vs_sweep.png,
figB_final_kl_vs_shots.png,meta.json}``.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ptqnn.backends import reference as ref
from ptqnn.backends.shots import ShotBackend
from ptqnn.tasks import bitcount
from ptqnn.train import kwsa

ROOT = Path(__file__).resolve().parents[1]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def git_hash() -> str:
    """HEAD, with a ``-dirty`` suffix when the tree has uncommitted changes (review 003 #3)."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=ROOT
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT
        ).stdout.strip()
        return f"{head}-dirty" if dirty else head
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_one(
    shots: int | None, crn: bool, seed: int, sweeps: int, n_bits: int, n_hidden: int
) -> dict[str, Any]:
    net = bitcount.network(n_bits, n_hidden)
    exact_loss = bitcount.kl_loss(net)  # monitor only
    if shots is None:
        backend: Any = ref.StatevectorBackend()
        loss = exact_loss
        ctx = None
    else:
        backend = ShotBackend(ref.StatevectorBackend(), shots=shots, seed=1000 + seed)
        loss = bitcount.kl_loss(net, backend)
        ctx = backend.common_random_numbers if crn else None
    cfg = kwsa.KWSAConfig(
        **{**bitcount.SUPERPOSITION_CONFIG.__dict__, "iterations": sweeps, "seed": seed}
    )
    x0 = net.random_params(np.random.default_rng(seed))
    exact_history = [exact_loss(x0)]

    def cb(n: int, x: np.ndarray, fx: float) -> None:
        exact_history.append(exact_loss(x))

    t0 = time.perf_counter()
    res = kwsa.minimise(loss, x0, cfg, callback=cb, probe_context=ctx)
    wall = time.perf_counter() - t0
    final = res.final_params if res.final_params is not None else res.params
    out = {
        "shots": shots,
        "crn": crn,
        "seed": seed,
        "exact_kl_history": [float(v) for v in exact_history],
        "exact_kl_final": float(exact_loss(final)),
        "exact_kl_best_iterate": float(min(exact_history)),
        "sampled_loss_final": float(res.final_loss),
        "evaluations": res.evaluations,
        "shots_used": int(getattr(backend, "shots_used", 0)),
        "wall_s": wall,
    }
    log(
        f"N={shots or 'exact':>6} crn={int(crn)} seed={seed}: exact KL {exact_history[0]:.4f} → "
        f"{out['exact_kl_final']:.5f} (best iterate {out['exact_kl_best_iterate']:.5f}) "
        f"{res.evaluations} evals, {wall:.1f} s"
    )
    return out


def plot_kl_vs_sweep(rows: list[dict[str, Any]], shots_list: list[int], out: Path) -> None:
    fig, axes = plt.subplots(1, len(shots_list), figsize=(4.2 * len(shots_list), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    exact = [r for r in rows if r["shots"] is None]
    for ax, n in zip(axes, shots_list, strict=True):
        for crn, color, label in (
            (False, "#d95f02", "independent draws"),
            (True, "#1b9e77", "common random numbers"),
        ):
            hs = np.array(
                [r["exact_kl_history"] for r in rows if r["shots"] == n and r["crn"] == crn]
            )
            med = np.median(hs, axis=0)
            ax.plot(med, color=color, lw=2, label=label)
            ax.fill_between(
                range(len(med)), hs.min(axis=0), hs.max(axis=0), color=color, alpha=0.18, lw=0
            )
        hs = np.array([r["exact_kl_history"] for r in exact])
        ax.plot(np.median(hs, axis=0), color="k", lw=1.2, ls="--", label="exact probabilities")
        ax.set_yscale("log")
        ax.set_title(f"N = {n} shots per evaluation")
        ax.set_xlabel("KW–SA sweep")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("exact KL(target ‖ model) of the iterate")
    axes[0].legend(fontsize=8)
    fig.suptitle("Bit-count training on sampled probabilities (median, min–max over 3 seeds)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_final_vs_shots(rows: list[dict[str, Any]], shots_list: list[int], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for crn, color, marker, label in (
        (False, "#d95f02", "o", "independent draws"),
        (True, "#1b9e77", "s", "common random numbers"),
    ):
        med = [
            np.median([r["exact_kl_final"] for r in rows if r["shots"] == n and r["crn"] == crn])
            for n in shots_list
        ]
        lo = [
            min(r["exact_kl_final"] for r in rows if r["shots"] == n and r["crn"] == crn)
            for n in shots_list
        ]
        hi = [
            max(r["exact_kl_final"] for r in rows if r["shots"] == n and r["crn"] == crn)
            for n in shots_list
        ]
        ax.errorbar(
            shots_list,
            med,
            yerr=[np.subtract(med, lo), np.subtract(hi, med)],
            color=color,
            marker=marker,
            capsize=3,
            lw=1.5,
            label=label,
        )
    exact_med = np.median([r["exact_kl_final"] for r in rows if r["shots"] is None])
    ax.axhline(exact_med, color="k", ls="--", lw=1.2, label="exact probabilities")
    k = 2 ** rows[0].get("n_bits", 5)
    ax.plot(
        shots_list,
        [(k - 1) / (2 * 0.487 * n) for n in shots_list],
        color="grey",
        ls=":",
        label="plug-in KL floor (K−1)/(2·P(out=1)·N)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("shots per loss evaluation N")
    ax.set_ylabel("exact KL after 20 sweeps")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def render(rows: list[dict[str, Any]], shots_list: list[int], meta: dict[str, Any]) -> str:
    L = [
        "# Bit-count trained on sampled probabilities — common random numbers for KW–SA\n",
        f"Run `{meta['run_dir']}` — git `{meta['git_hash'][:12]}` — "
        f"wall {meta['wall_seconds']:.0f} s — "
        f"`Network({meta['n_bits']}, {meta['n_hidden']})`, {meta['sweeps']} sweeps of our "
        "reproduction schedule "
        "(`bitcount.SUPERPOSITION_CONFIG`), 3 optimiser seeds; the loss is the plug-in KL of "
        "`counts/N`; the column values are the **exact** KL of the final iterate.\n",
        "| N | CRN | exact KL final (median, min–max) | best iterate (median) | shots per run |",
        "|---:|---|---|---:|---:|",
    ]
    ex = [r["exact_kl_final"] for r in rows if r["shots"] is None]
    L.append(
        f"| ∞ | — | {np.median(ex):.5f} ({min(ex):.5f}–{max(ex):.5f}) | "
        f"{np.median([r['exact_kl_best_iterate'] for r in rows if r['shots'] is None]):.5f} "
        "| — |"
    )
    for n in shots_list:
        for crn in (False, True):
            sel = [r for r in rows if r["shots"] == n and r["crn"] == crn]
            f = [r["exact_kl_final"] for r in sel]
            L.append(
                f"| {n} | {'on' if crn else 'off'} | "
                f"{np.median(f):.5f} ({min(f):.5f}–{max(f):.5f}) | "
                f"{np.median([r['exact_kl_best_iterate'] for r in sel]):.5f} | "
                f"{sel[0]['shots_used']:,} |"
            )
    L += ["", "![A](figA_kl_vs_sweep.png)", "", "![B](figB_final_kl_vs_shots.png)", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--shots", type=int, nargs="+", default=[1000, 4000, 16000])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--sweeps", type=int, default=20)
    ap.add_argument("--n-bits", type=int, default=5)
    ap.add_argument("--n-hidden", type=int, default=5)
    ap.add_argument(
        "--budget-s",
        type=float,
        default=900.0,
        help="abort if the projected wall time exceeds this",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/runs/bitcount_shots")
    args = ap.parse_args(argv)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out / stamp
    out.mkdir(parents=True, exist_ok=False)
    log(f"run dir {out}")
    t_start = time.perf_counter()

    jobs: list[tuple[int | None, bool]] = [(None, False)] + [
        (n, c) for n in args.shots for c in (False, True)
    ]
    rows: list[dict[str, Any]] = []
    first = run_one(jobs[0][0], jobs[0][1], args.seeds[0], args.sweeps, args.n_bits, args.n_hidden)
    rows.append(first)
    projected = first["wall_s"] * len(jobs) * len(args.seeds) * 1.5  # sampling overhead margin
    log(f"projected wall ≈ {projected:.0f} s for {len(jobs) * len(args.seeds)} runs")
    if projected > args.budget_s:
        raise SystemExit(
            f"projected {projected:.0f} s exceeds --budget-s {args.budget_s:.0f}; aborting"
        )
    for shots, crn in jobs:
        for seed in args.seeds:
            if (shots, crn, seed) == (jobs[0][0], jobs[0][1], args.seeds[0]):
                continue
            rows.append(run_one(shots, crn, seed, args.sweeps, args.n_bits, args.n_hidden))
    for r in rows:
        r["n_bits"] = args.n_bits

    meta = {
        "experiment": "bitcount_shots",
        "run_dir": str(out.relative_to(ROOT)),
        "timestamp_utc": stamp,
        "git_hash": git_hash(),
        "backend": "numpy-statevector (+ShotBackend)",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "argv": sys.argv[1:],
        "n_bits": args.n_bits,
        "n_hidden": args.n_hidden,
        "sweeps": args.sweeps,
        "seeds": args.seeds,
        "shots": args.shots,
        "kwsa": bitcount.SUPERPOSITION_CONFIG.__dict__,
        "wall_seconds": time.perf_counter() - t_start,
    }
    (out / "results.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    plot_kl_vs_sweep(rows, args.shots, out / "figA_kl_vs_sweep.png")
    plot_final_vs_shots(rows, args.shots, out / "figB_final_kl_vs_shots.png")
    (out / "results.md").write_text(render(rows, args.shots, meta))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    log(f"done in {meta['wall_seconds']:.0f} s → {out / 'results.md'}")
    return out


if __name__ == "__main__":
    main()
