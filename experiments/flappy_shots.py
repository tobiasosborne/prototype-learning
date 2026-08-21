"""Flappy Bird under shot noise: margin-aware training and sequential decisions.
(Shot-noise programme, items 1 and 3.)

`reports/noise.md` found that the trained policy's decisions sit a median δ = 0.019 from the
flap threshold, so a frame decided from ``N`` shots flips with probability ≈ Φ(−2δ√N) and perfect
play needs ≳ 10⁴ shots per frame.  Two remedies, measured here by *playing the game*:

1. **Margin-aware training.**  Fine-tune the trained policy with a per-frame penalty
   ``w · 0.01 · max(0, 1 − δ/m)`` added to the shaped episode return
   (``ObjectiveConfig(margin=m, margin_weight=w)``; 25 sweeps from the paper-style parameters,
   5 training seeds).  Shots scale as ``1/δ²``, so pushing the visited margins from 0.02 to 0.1
   is ×25 fewer shots — if the game can be played with such a policy at all.  (Training from
   scratch with a *mean* hinge of weight 20 collapsed to a confidently dead policy in the first
   attempt — run ``20260821T075048Z`` — which is why the penalty is per frame and below the
   per-frame survival reward.)
2. **Sequential decisions (SPRT).**  Instead of a fixed ``N`` per frame, Wald's sequential test on
   the sign of ``p − ½`` with indifference margin ``δ0`` and cap ``N_max``
   (:class:`ptqnn.tasks.flappy.agent.SPRTDecision`): frames far from the boundary stop after tens
   of shots, only the near-boundary frames pay the cap.

Both policies (paper-style, margin-aware) are then played on the 20 held-out seeds 5..24 (5000
frames = 110 pipes is a perfect game) with fixed ``N ∈ {250, …, 10000}`` and with SPRT rules,
recording perfect-game fraction, score, flips per frame against the exact decision, and the mean
shots per frame actually spent.

    uv run python experiments/flappy_shots.py            # ≈ 3 min training + ≈ 2 min play

writes ``experiments/runs/flappy_shots/<stamp>/{results.json,results.md,fig*.png,meta.json,
params_margin.npy}``.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ptqnn.circuits.neurons import Network
from ptqnn.tasks.flappy import agent as ag
from ptqnn.tasks.flappy import train as tr
from ptqnn.train import kwsa

ROOT = Path(__file__).resolve().parents[1]
BASELINE_RUN = ROOT / "experiments/runs/flappy/20260820T143753Z"
EVAL_SEEDS = tuple(range(5, 25))
PERFECT = 110


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


def train_margin_aware(
    net: Network, x0: np.ndarray, margin: float, weight: float, seed: int, iterations: int
) -> tuple[np.ndarray, dict[str, Any]]:
    oc = tr.ObjectiveConfig(margin=margin, margin_weight=weight)
    cfg = kwsa.KWSAConfig(**{**asdict(tr.DEFAULT_KWSA), "iterations": iterations, "seed": seed})
    t0 = time.perf_counter()

    def cb(n: int, x: np.ndarray, fx: float) -> None:
        if n % 5 == 0 or n == iterations - 1:
            log(f"  margin-aware (w={weight}) sweep {n:2d}: loss {fx: .3f}")

    res = tr.train(
        net,
        cfg=cfg,
        objective_config=oc,
        eval_seeds=EVAL_SEEDS,
        eval_steps=5000,
        callback=cb,
        x0=x0,
    )
    wall = time.perf_counter() - t0
    info = {
        "margin": margin,
        "margin_weight": weight,
        "seed": seed,
        "iterations": iterations,
        "init_loss": res.init_loss,
        "final_loss": res.loss,
        "evaluations": res.evaluations,
        "scores_exact": res.scores,
        "perfect_exact": float(np.mean([s >= PERFECT for s in res.scores])),
        "wall_s": wall,
    }
    log(
        f"margin-aware training done: loss {res.init_loss:.2f} → {res.loss:.2f}, "
        f"{res.evaluations} evals, exact play {info['perfect_exact']:.0%} perfect, {wall:.0f} s"
    )
    return res.params, info


def play(
    net: Network,
    params: np.ndarray,
    shots: int,
    decision: ag.SPRTDecision | None,
    label: str,
) -> dict[str, Any]:
    """20 held-out episodes with the given decision rule; flips are counted against the exact
    decision of the same frame."""
    oc = tr.ObjectiveConfig()
    exact = tr.make_policy(net, params, oc)
    scores, steps, flips, frames, shots_used = [], [], 0, 0, 0
    margins: list[float] = []
    t0 = time.perf_counter()
    for i, seed in enumerate(EVAL_SEEDS):
        pol = tr.make_policy(
            net, params, oc, shots=shots, shot_seed=1000 * i + 7, decision=decision
        )
        # wrap act() to compare with the exact decision on every frame
        inner_act = pol.act

        def act(obs: np.ndarray, _inner: Any = inner_act) -> bool:
            nonlocal flips
            d = _inner(obs)
            if d != exact.act(obs):
                flips += 1
            return d

        pol.act = act  # type: ignore[method-assign]
        env = tr.FlappyEnv(config=oc.env_config, seed=seed, features=oc.features)
        res = ag.play_episode(env, pol, 5000)
        scores.append(int(res.score))
        steps.append(int(res.steps))
        frames += len(pol.frame_margins)
        margins += pol.frame_margins
        shots_used += pol.shots_used
    out = {
        "label": label,
        "shots": shots,
        "sprt": None if decision is None else asdict(decision),
        "scores": scores,
        "median_score": float(np.median(scores)),
        "perfect": float(np.mean([s >= PERFECT for s in scores])),
        "frames": frames,
        "flips_per_frame": flips / frames if frames else 0.0,
        "mean_shots_per_frame": shots_used / frames if frames else 0.0,
        "margin_quantiles": {
            str(q): float(np.quantile(margins, q)) for q in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9)
        },
        "wall_s": time.perf_counter() - t0,
    }
    log(
        f"{label:>12} shots={shots:>6} "
        f"{'SPRT δ0=' + str(decision.delta0) if decision else 'fixed':>12}: "
        f"perfect {out['perfect']:.0%}, median {out['median_score']:.0f} pipes, "
        f"flips/frame {out['flips_per_frame']:.3%}, shots/frame {out['mean_shots_per_frame']:.0f} "
        f"({out['wall_s']:.0f} s)"
    )
    return out


# --- figures -------------------------------------------------------------------------------------


COLORS = ("#7570b3", "#1b9e77", "#d95f02", "#e7298a", "#66a61e")


def _labels(rows: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for r in rows:
        if r["label"] not in seen:
            seen.append(r["label"])
    return seen


def fig_margins(rows: list[dict[str, Any]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    qs = ["0.01", "0.1", "0.25", "0.5", "0.75", "0.9"]
    for label, color in zip(_labels(rows), COLORS, strict=False):
        r = next(r for r in rows if r["label"] == label and r["shots"] == 0)
        ax.plot(
            [float(q) for q in qs],
            [r["margin_quantiles"][q] for q in qs],
            marker="o",
            color=color,
            label=label,
        )
    for n, ls in ((1000, ":"), (4000, "--"), (10000, "-.")):
        ax.axhline(
            1 / (2 * np.sqrt(n)), color="grey", ls=ls, lw=1, label=f"δ resolved by N={n} (1/2√N)"
        )
    ax.set_xlabel("quantile of visited frames")
    ax.set_ylabel("decision margin δ = |P(out=1) − ½|")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)
    ax.set_title("Decision margins over played frames")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_perfect_vs_shots(rows: list[dict[str, Any]], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for label, color in zip(_labels(rows), COLORS, strict=False):
        fixed = sorted(
            (r for r in rows if r["label"] == label and r["shots"] > 0 and r["sprt"] is None),
            key=lambda r: r["shots"],
        )
        axes[0].plot(
            [r["shots"] for r in fixed],
            [100 * r["perfect"] for r in fixed],
            marker="o",
            color=color,
            label=f"{label}, fixed N",
        )
        axes[1].plot(
            [r["shots"] for r in fixed],
            [100 * r["flips_per_frame"] for r in fixed],
            marker="o",
            color=color,
            label=f"{label}, fixed N",
        )
        sprt = [r for r in rows if r["label"] == label and r["sprt"] is not None]
        axes[0].scatter(
            [r["mean_shots_per_frame"] for r in sprt],
            [100 * r["perfect"] for r in sprt],
            marker="*",
            s=120,
            color=color,
            edgecolor="k",
            zorder=5,
            label=f"{label}, SPRT (x = mean shots/frame)",
        )
        axes[1].scatter(
            [r["mean_shots_per_frame"] for r in sprt],
            [100 * r["flips_per_frame"] for r in sprt],
            marker="*",
            s=120,
            color=color,
            edgecolor="k",
            zorder=5,
        )
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("shots per frame")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("perfect games (%)")
    axes[1].set_ylabel("frames decided differently from exact (%)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Shot budget vs play quality (20 held-out seeds)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def render(rows: list[dict[str, Any]], train_info: dict[str, Any], meta: dict[str, Any]) -> str:
    L = [
        "# Flappy Bird under shot noise — margin-aware training and sequential decisions\n",
        f"Run `{meta['run_dir']}` — git `{meta['git_hash'][:12]}` — "
        f"wall {meta['wall_seconds']:.0f} s.  "
        f"Paper-style parameters from `{BASELINE_RUN.relative_to(ROOT)}`; margin-aware "
        f"parameters fine-tuned from them here (m = {train_info['margin']}; per weight: "
        + "; ".join(
            f"w={t['margin_weight']:g}: loss {t['init_loss']:.2f} → {t['final_loss']:.2f}, "
            f"{t['evaluations']} evals, exact play {t['perfect_exact']:.0%} perfect"
            for t in train_info["runs"]
        )
        + ").  20 held-out seeds 5..24, 5000 frames.\n",
        "| policy | decision rule | perfect | median pipes | flips/frame | mean shots/frame "
        "| δ p10 | δ p50 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        rule = (
            "exact"
            if r["shots"] == 0 and r["sprt"] is None
            else (
                f"SPRT δ0={r['sprt']['delta0']}, cap {r['sprt']['max_shots']}"
                if r["sprt"]
                else f"fixed N={r['shots']}"
            )
        )
        L.append(
            f"| {r['label']} | {rule} | {r['perfect']:.0%} | {r['median_score']:.0f} | "
            f"{r['flips_per_frame']:.2%} | {r['mean_shots_per_frame']:.0f} | "
            f"{r['margin_quantiles']['0.1']:.4f} | {r['margin_quantiles']['0.5']:.4f} |"
        )
    L += ["", "![margins](figA_margins.png)", "", "![shots](figB_perfect_vs_shots.png)", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--margin-weight", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iterations", type=int, default=25)
    ap.add_argument("--shots", type=int, nargs="+", default=[250, 500, 1000, 2000, 4000, 10000])
    ap.add_argument("--sprt-delta0", type=float, nargs="+", default=[0.005, 0.01, 0.02])
    ap.add_argument("--sprt-cap", type=int, nargs="+", default=[4000, 10000])
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/runs/flappy_shots")
    args = ap.parse_args(argv)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out / stamp
    out.mkdir(parents=True, exist_ok=False)
    log(f"run dir {out}")
    t_start = time.perf_counter()

    net = Network(n_inputs=2, n_hidden=4, hidden="prototype")
    base = np.load(BASELINE_RUN / "params.npy")
    policies: list[tuple[str, np.ndarray]] = [("paper-style", base)]
    train_infos: list[dict[str, Any]] = []
    for w in args.margin_weight:
        mp, info = train_margin_aware(net, base, args.margin, w, args.seed, args.iterations)
        np.save(out / f"params_margin_w{w:g}.npy", mp)
        policies.append((f"margin w={w:g}", mp))
        train_infos.append(info)
    train_info = {"runs": train_infos, "margin": args.margin}

    rows: list[dict[str, Any]] = []
    for label, params in policies:
        rows.append(play(net, params, 0, None, label))
        for n in args.shots:
            rows.append(play(net, params, n, None, label))
        for cap in args.sprt_cap:
            for d0 in args.sprt_delta0:
                rows.append(
                    play(net, params, cap, ag.SPRTDecision(delta0=d0, max_shots=cap), label)
                )

    meta = {
        "experiment": "flappy_shots",
        "run_dir": str(out.relative_to(ROOT)),
        "timestamp_utc": stamp,
        "git_hash": git_hash(),
        "backend": "numpy-statevector (basis columns) + binomial / SPRT decisions",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "argv": sys.argv[1:],
        "baseline_params": str(BASELINE_RUN.relative_to(ROOT) / "params.npy"),
        "eval_seeds": list(EVAL_SEEDS),
        "train": train_info,
        "wall_seconds": time.perf_counter() - t_start,
    }
    (out / "results.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    fig_margins(rows, out / "figA_margins.png")
    fig_perfect_vs_shots(rows, out / "figB_perfect_vs_shots.png")
    (out / "results.md").write_text(render(rows, train_info, meta))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    log(f"done in {meta['wall_seconds']:.0f} s → {out / 'results.md'}")
    return out


if __name__ == "__main__":
    main()
