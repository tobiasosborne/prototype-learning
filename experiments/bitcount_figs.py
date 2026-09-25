"""Reproduce PTQNN bit-count Figs 3, 5, 6 from the trained modules — no new physics/ML here.

``[PTQNN Fig 3]``: "activation probability of the prototype |01⟩: If |q0⟩ is in 0 and |q1⟩ is
in 1, the probability of activation is maximized. If only one condition is met, the activation
is reduced and if the pattern does not match, the activation is zero" — a single prototype
neuron, ``prototype=(π, 0)``, ``theta=π/2`` on inputs prepared by ``RX(π·q0)``, ``RX(π·q1)``.

``[PTQNN Fig 5, §III.C]``: bit-count network (5 inputs, 5 hidden prototype neurons, 1 classic
output — 11 qubits), trained under superposition (``bitcount.train_superposition``) with the
histogram of ``P(inputs = x | out = 1)`` compared to the target ``(popcount(x)+1)/112``.

``[PTQNN Fig 6]``: convergence of the superposition scheme (KL ≈0.055 → ≈0 within ~8
iterations) against the sample-wise iterative baseline (``bitcount.train_iterative``, ~120
iterations to converge), across seeds as a median + min–max band.

All physics/training logic lives in ``ptqnn.tasks.bitcount``, ``ptqnn.train.kwsa``,
``ptqnn.train.distribution`` and ``ptqnn.circuits.neurons`` — this script only drives them and
plots. Threading is pinned to 1 OS thread per process *before numpy is imported anywhere* so
that a ``multiprocessing`` pool over seeds gets full parallelism instead of BLAS oversubscription.
"""

from __future__ import annotations

import os

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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ptqnn.backends import reference as ref
from ptqnn.circuits import ir
from ptqnn.circuits.neurons import prototype_neuron
from ptqnn.tasks import bitcount
from ptqnn.train import distribution, kwsa

N_BITS = 5
N_HIDDEN = 5  # [PTQNN Fig 5]: 5 hidden prototype neurons, 11 qubits total.

IterativeMonitor = Literal["perbasis", "superposition"]

# --- Fig 3 -------------------------------------------------------------------------------------

# [PTQNN Fig 3 caption]: "leads to the highest activation for a [1 0] input prototype" (paper's
# 1-based, top-to-bottom qubit numbering) == our qubit 0 = 1, qubit 1 = 0 gives the *lowest*
# activation and qubit0=0, qubit1=1 the *highest* — the caption names the prototype |01⟩, i.e.
# ``prototype=(π, 0)`` so RX(π) on input 0's rotation and RX(0) on input 1's before the CRX.
FIG3_PROTOTYPE = (np.pi, 0.0)
FIG3_THETA = np.pi / 2
FIG3_GRID_N = 41


def fig3_activation_grid(n: int = FIG3_GRID_N) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``P(out=1)`` for a single prototype neuron over ``q0, q1 ∈ [0,1]`` on an ``n×n`` grid.

    Inputs are prepared by ``RX(π·q0)``, ``RX(π·q1)`` [task spec]; the neuron is
    ``neurons.prototype_neuron((0,1), 2, prototype=(π,0), theta=π/2)`` on a 3-qubit circuit,
    evaluated exactly with the statevector backend.
    """
    qs = np.linspace(0.0, 1.0, n)
    backend = ref.StatevectorBackend()
    grid = np.zeros((n, n), dtype=np.float64)
    for i, q0 in enumerate(qs):
        for j, q1 in enumerate(qs):
            circ = ir.Circuit(3)
            circ.append(ir.rx(0, np.pi * q0))
            circ.append(ir.rx(1, np.pi * q1))
            circ.extend(prototype_neuron((0, 1), 2, FIG3_PROTOTYPE, FIG3_THETA))
            p = backend.probabilities(circ)
            grid[i, j] = ref.marginal(p, [2])[1]
    return qs, qs, grid


def fig3_corner_check(qs0: np.ndarray, qs1: np.ndarray, grid: np.ndarray) -> dict[str, Any]:
    """Corners of the grid against ``[PTQNN Fig 3]``'s qualitative description."""
    corners = {
        "(q0=0, q1=0)": (float(grid[0, 0]), 0.5),
        "(q0=0, q1=1)": (float(grid[0, -1]), 1.0),
        "(q0=1, q1=0)": (float(grid[-1, 0]), 0.0),
        "(q0=1, q1=1)": (float(grid[-1, -1]), 0.5),
    }
    passed = all(abs(v - e) < 1e-6 for v, e in corners.values())
    return {"corners": corners, "passed": passed}


def plot_fig3(qs0: np.ndarray, qs1: np.ndarray, grid: np.ndarray, out_path: Path) -> None:
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(projection="3d")
    q0_mesh, q1_mesh = np.meshgrid(qs0, qs1, indexing="ij")
    surf = ax.plot_surface(q0_mesh, q1_mesh, grid, cmap="jet", edgecolor="none", antialiased=True)
    ax.set_xlabel("$q_0$")
    ax.set_ylabel("$q_1$")
    ax.set_zlabel("activation probability")
    ax.set_title("Fig 3: activation probability of the prototype |01⟩")
    fig.colorbar(surf, shrink=0.6, aspect=12, pad=0.1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --- Fig 5 / Fig 6 -------------------------------------------------------------------------------


def _train_seed(args: tuple[int, int, int, IterativeMonitor, int, int]) -> dict[str, Any]:
    """Worker: superposition + iterative training for one seed. Runs in its own process.

    Review 004 B2: ``*_loss`` / ``*_params`` are the best point seen (``KWSAResult.loss``; for
    the iterative scheme the best *monitored* point), ``*_final`` / ``*_final_params`` the last
    state (``history[-1]``).  The two differ — annealing accepts uphill moves and the
    one-example iterative objective is non-stationary — so both are carried.
    """
    seed, iterations, iterative_iterations, monitor, batch, monitor_every = args
    net = bitcount.network(N_BITS, N_HIDDEN)

    t0 = time.perf_counter()
    res_s = bitcount.train_superposition(net, seed=seed, iterations=iterations)
    t1 = time.perf_counter()
    res_i = bitcount.train_iterative(
        net,
        seed=seed,
        iterations=iterative_iterations,
        monitor=monitor,
        batch=batch or None,
        monitor_every=monitor_every,
    )
    t2 = time.perf_counter()

    return {
        "seed": seed,
        "superposition_loss": float(res_s.loss),
        "superposition_final": float(res_s.history[-1]),
        "superposition_history": [float(v) for v in res_s.history],
        "superposition_params": res_s.params.tolist(),
        "superposition_final_params": _final_params(res_s).tolist(),
        "superposition_evaluations": res_s.evaluations,
        "superposition_wall_s": t1 - t0,
        "iterative_monitor": monitor,
        "iterative_batch": batch,
        "iterative_monitor_every": monitor_every,
        "iterative_iterations": iterative_iterations,
        "iterative_loss": float(res_i.loss),
        "iterative_final": float(res_i.history[-1]),
        "iterative_history": [float(v) for v in res_i.history],
        "iterative_params": res_i.params.tolist(),
        "iterative_final_params": _final_params(res_i).tolist(),
        "iterative_extra_history": {
            key: [float(v) for v in vals] for key, vals in res_i.extra_history.items()
        },
        "iterative_evaluations": res_i.evaluations,
        "iterative_wall_s": t2 - t1,
    }


def _final_params(res: kwsa.KWSAResult) -> np.ndarray:
    if res.final_params is None:  # pragma: no cover - both trainers always set it
        raise ValueError("training result carries no final parameters")
    return res.final_params


SCHEMES = ("superposition", "iterative")


def final_kl(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """``{seed: {scheme: last recorded KL}}`` — the value a run *ends* at (review 004 B2)."""
    return {str(r["seed"]): {s: r[f"{s}_final"] for s in SCHEMES} for r in results}


def best_kl(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """``{seed: {scheme: best KL seen}}`` — the minimum over the run, not its end point."""
    return {str(r["seed"]): {s: r[f"{s}_loss"] for s in SCHEMES} for r in results}


def kl_table_lines(results: list[dict[str, Any]]) -> list[str]:
    """Markdown table of final and best KL per seed and scheme (results.md)."""
    lines = [
        "| seed | superposition final KL | superposition best KL "
        "| iterative final KL | iterative best KL |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['seed']} | {r['superposition_final']:.6f} | {r['superposition_loss']:.6f} "
            f"| {r['iterative_final']:.6f} | {r['iterative_loss']:.6f} |"
        )
    return lines


def save_params(out_dir: Path, results: list[dict[str, Any]]) -> list[str]:
    """Archive the trained parameters per seed and scheme so diagnostics can be recomputed
    (review 004 M8): ``params-<scheme>-<seed>.npy`` is the final state (the network of
    ``final_kl``), ``params-<scheme>-best-<seed>.npy`` the best one (of ``best_kl``; for
    superposition the Fig-5 network).  Returns the file names written."""
    written: list[str] = []
    for r in results:
        for scheme in SCHEMES:
            for name, key in (
                (f"params-{scheme}-{r['seed']}.npy", f"{scheme}_final_params"),
                (f"params-{scheme}-best-{r['seed']}.npy", f"{scheme}_params"),
            ):
                np.save(out_dir / name, np.asarray(r[key], dtype=np.float64))
                written.append(name)
    return written


def provenance(pool_size: int) -> dict[str, Any]:
    """Which machine produced the run (review 004 M7/B3)."""
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "pool_size": pool_size,
    }


def plot_fig5(best: dict[str, Any], out_path: Path) -> None:
    net = bitcount.network(N_BITS, N_HIDDEN)
    params = np.array(best["superposition_params"], dtype=np.float64)
    backend = ref.StatevectorBackend()
    model = distribution.conditioned_input_distribution(
        net, params, bitcount.superposition_prep(net), backend
    )
    target = bitcount.target_distribution(N_BITS)
    labels = [format(x, f"0{N_BITS}b") for x in range(2**N_BITS)]

    fig, (ax_model, ax_target) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax_model.bar(labels, model, color="#4c72b0", edgecolor="black", linewidth=0.3)
    ax_model.set_ylabel("Probability")
    ax_model.set_title(
        f"Fig 5: trained model P(x | out=1) — seed {best['seed']}, "
        f"best KL={best['superposition_loss']:.4g}"
    )
    ax_target.bar(labels, target, color="#4c72b0", edgecolor="black", linewidth=0.3)
    ax_target.set_ylabel("Probability")
    ax_target.set_title("Fig 5: target distribution p(x) = (popcount(x)+1)/112")
    ax_target.set_xticks(range(len(labels)))
    ax_target.set_xticklabels(labels, rotation=90, fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fig6(results: list[dict[str, Any]], out_path: Path) -> tuple[np.ndarray, np.ndarray]:
    sup = np.array([r["superposition_history"] for r in results])  # (n_seeds, iterations+1)
    itr = np.array([r["iterative_history"] for r in results])  # (n_seeds, iterative_iterations+1)

    sup_med, sup_lo, sup_hi = np.median(sup, axis=0), sup.min(axis=0), sup.max(axis=0)
    itr_med, itr_lo, itr_hi = np.median(itr, axis=0), itr.min(axis=0), itr.max(axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    # history index i ↔ iteration i·monitor_every, except the final record, which sits at the
    # last iteration even off the cadence (review 004 M2)
    x_itr = bitcount.monitor_x(
        itr.shape[1],
        int(results[0]["iterative_monitor_every"]),
        int(results[0]["iterative_iterations"]),
    )
    x_sup = np.arange(sup.shape[1])
    ax.fill_between(x_itr, itr_lo, itr_hi, color="red", alpha=0.2, linewidth=0)
    mon = results[0]["iterative_monitor"]
    ax.plot(x_itr, itr_med, color="red", linewidth=2, label=f"Iterative Optimization ({mon})")
    extra_key = next(iter(results[0]["iterative_extra_history"]), None)
    if extra_key is not None:
        extra = np.array([r["iterative_extra_history"][extra_key] for r in results])
        ax.plot(
            x_itr,
            np.median(extra, axis=0),
            color="red",
            linewidth=1,
            linestyle="--",
            label=f"Iterative ({extra_key}, secondary)",
        )
    ax.fill_between(x_sup, sup_lo, sup_hi, color="blue", alpha=0.2, linewidth=0)
    ax.plot(x_sup, sup_med, color="blue", linewidth=2, label="Optimization using superposition")
    if itr.shape[1] > 10 * sup.shape[1]:
        # 20 superposition sweeps against 1e5 elementary steps: symlog keeps both readable
        ax.set_xscale("symlog", linthresh=float(sup.shape[1]))
    ax.set_xlabel("# Iterations")
    ax.set_ylabel("KL divergence")
    ax.set_title("Fig 6: convergence (median + min–max band over seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return sup_med, itr_med


# --- driver --------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_out = "experiments/runs/bitcount/" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--iterative-iterations", type=int, default=200)
    p.add_argument(
        "--iterative-monitor",
        choices=["perbasis", "superposition"],
        default="perbasis",
        help="KL series for the iterative (red) curve; the other is plotted dashed.",
    )
    p.add_argument(
        "--iterative-batch",
        type=int,
        default=0,
        help="distinct basis inputs per iterative step; 0 = single example, 32 = full batch.",
    )
    p.add_argument("--iterative-monitor-every", type=int, default=1)
    p.add_argument("--out", type=str, default=default_out)
    return p.parse_args(argv)


def git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    wall_t0 = time.perf_counter()

    # --- Fig 3 -----------------------------------------------------------------------------
    print("[fig3] computing activation grid...")
    t0 = time.perf_counter()
    qs0, qs1, grid = fig3_activation_grid(FIG3_GRID_N)
    fig3_wall = time.perf_counter() - t0
    check = fig3_corner_check(qs0, qs1, grid)
    print("[fig3] corners:")
    for name, (val, expected) in check["corners"].items():
        print(f"    {name}: {val:.6f}  (expected ~{expected})")
    print(f"[fig3] corner check passed: {check['passed']}")
    plot_fig3(qs0, qs1, grid, out_dir / "fig03_activation.png")
    np.save(out_dir / "fig03_grid.npy", grid)

    # --- Fig 5 / Fig 6: train all seeds in parallel ------------------------------------------
    print(
        f"[train] {len(args.seeds)} seeds x ({args.iterations} superposition + "
        f"{args.iterative_iterations} iterative iterations), pool over seeds..."
    )
    t0 = time.perf_counter()
    ctx = mp.get_context("fork")
    monitor = cast(IterativeMonitor, args.iterative_monitor)
    jobs = [
        (
            seed,
            args.iterations,
            args.iterative_iterations,
            monitor,
            args.iterative_batch,
            args.iterative_monitor_every,
        )
        for seed in args.seeds
    ]
    pool_size = max(1, min(len(args.seeds), os.cpu_count() or 1))
    with ctx.Pool(processes=pool_size) as pool:
        results = pool.map(_train_seed, jobs)
    train_wall = time.perf_counter() - t0
    results.sort(key=lambda r: r["seed"])
    params_files = save_params(out_dir, results)

    print("[fig5/6] KL per seed (final = last recorded value, best = minimum seen):")
    for line in kl_table_lines(results):
        print("    " + line)

    best = min(results, key=lambda r: r["superposition_loss"])
    print(f"[fig5] best seed = {best['seed']} (best KL = {best['superposition_loss']:.6f})")
    plot_fig5(best, out_dir / "fig05_histograms.png")

    sup_med, itr_med = plot_fig6(results, out_dir / "fig06_convergence.png")
    below = np.flatnonzero(sup_med < 0.005)
    first_below = int(below[0]) if below.size else None
    print(
        f"[fig6] median superposition KL first < 0.005 at iteration "
        f"{first_below if first_below is not None else '(never)'}"
    )

    with (out_dir / "histories.json").open("w") as f:
        histories: dict[str, Any] = {
            "superposition": {str(r["seed"]): r["superposition_history"] for r in results},
            "iterative": {str(r["seed"]): r["iterative_history"] for r in results},
            "iterative_monitor": results[0]["iterative_monitor"],
        }
        secondary = next(iter(results[0]["iterative_extra_history"]), None)
        if secondary is not None:
            histories[f"iterative_{secondary}"] = {
                str(r["seed"]): r["iterative_extra_history"][secondary] for r in results
            }
        json.dump(histories, f, indent=2)

    total_wall = time.perf_counter() - wall_t0

    # --- meta.json ---------------------------------------------------------------------------
    meta = {
        "seeds": args.seeds,
        "iterations": args.iterations,
        "iterative_iterations": args.iterative_iterations,
        "iterative_monitor": args.iterative_monitor,
        "iterative_batch": args.iterative_batch,
        "iterative_monitor_every": args.iterative_monitor_every,
        "n_bits": N_BITS,
        "n_hidden": N_HIDDEN,
        "fig3_grid_n": FIG3_GRID_N,
        "git_hash": git_hash(),
        "backend": ref.StatevectorBackend().name,
        **provenance(pool_size),
        "wall_times_s": {
            "fig3": fig3_wall,
            "train_pool_total": train_wall,
            "total": total_wall,
            "per_seed": {
                str(r["seed"]): {
                    "superposition": r["superposition_wall_s"],
                    "iterative": r["iterative_wall_s"],
                }
                for r in results
            },
        },
        "fig3_corner_check": check,
        # review 004 B2: final_kl is the last recorded value (it used to be the minimum)
        "final_kl": final_kl(results),
        "best_kl": best_kl(results),
        "params_files": params_files,
        "best_seed": best["seed"],
        "median_superposition_kl_below_0.005_at_iteration": first_below,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    with (out_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    # --- results.md ----------------------------------------------------------------------------
    lines = [
        "# Bit-count reproduction — Figs 3, 5, 6",
        "",
        f"Run: `{out_dir}` — git `{meta['git_hash'][:12]}` — backend `{meta['backend']}` — "
        f"seeds {args.seeds} — wall time {total_wall:.1f}s",
        "",
        "## Fig 3 — prototype activation surface",
        "",
        "Corner check (single prototype neuron, prototype=(π,0), theta=π/2, "
        "[PTQNN Fig 3 caption]):",
        "",
        "| corner (q0,q1) | value | expected |",
        "|---|---|---|",
    ]
    for name, (val, expected) in check["corners"].items():
        lines.append(f"| {name} | {val:.6f} | ~{expected} |")
    lines += [
        f"\nPassed: **{check['passed']}**",
        "",
        "## Fig 5 — bit-count histograms",
        "",
        "Final = last recorded KL; best = minimum seen during the run (for the iterative scheme,",
        "over the monitored points).  Parameters of both: `params-<scheme>[-best]-<seed>.npy`.",
        "",
        *kl_table_lines(results),
    ]
    lines += [
        f"\nBest seed: **{best['seed']}** (best KL = {best['superposition_loss']:.6f}) — histogram "
        f"plotted against target `p(x) = (popcount(x)+1)/112`.",
        "",
        "## Fig 6 — convergence",
        "",
        "Paper [PTQNN Fig 6]: superposition KL ≈0.055 → ≈0 within ~8 iterations; iterative "
        "sample-wise baseline ≈120 iterations.",
        "",
        f"This run: median superposition KL first < 0.005 at iteration "
        f"**{first_below if first_below is not None else 'never'}** "
        f"(out of {args.iterations}); median superposition KL at iteration 0 = "
        f"{sup_med[0]:.4f}, at last iteration = {sup_med[-1]:.4f}. Median iterative KL at "
        f"iteration 0 = {itr_med[0]:.4f}, at last iteration ({args.iterative_iterations}) = "
        f"{itr_med[-1]:.4f}.",
        "",
    ]
    if itr_med[-1] > 0.01:
        lines += [
            "**Discrepancy** (reported, not tuned away — `CLAUDE.md` ground-truth discipline): "
            "the paper's Fig 6 (red curve) reaches KL ≈ 0 by ~120 iterations; this run's "
            f"iterative baseline (`bitcount.train_iterative`) only reaches median KL "
            f"{itr_med[-1]:.4f} after {args.iterative_iterations} iterations — it is qualitatively "
            "slower than superposition training (matching the paper) but does not converge to "
            "near-zero on this budget. The scheme's design — one elementary KW step on one "
            "coordinate, graded on the per-input activation distribution and annealed once per "
            "nominal epoch of 2**n sampled inputs (`[session-4 analysis]` (a)–(c)) — is "
            "itself `[ASSUMPTION]` (`ptqnn/tasks/bitcount.py` docstring, `PLAN.md §0` "
            "'Iterative baseline of Fig 6'); the paper gives no hyperparameters for it.",
            "",
        ]
    (out_dir / "results.md").write_text("\n".join(lines) + "\n")

    print(f"[done] wrote {out_dir} in {total_wall:.1f}s")


if __name__ == "__main__":
    main()
