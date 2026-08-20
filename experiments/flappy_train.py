"""Train the Flappy Bird agent of ``[PTQNN §III.D, Fig 10]`` and record the run.

    uv run python experiments/flappy_train.py --seed 0 --iterations 25

Writes to ``experiments/runs/flappy/<timestamp>/``: ``params.npy`` (the 17 trained angles),
``meta.json`` (seed, git hash, every config), ``log.csv`` (loss and evaluation score per KW–SA
iteration) and ``best.gif`` (the first frames of the best episode).
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ptqnn.backends import reference as ref
from ptqnn.circuits.neurons import Network
from ptqnn.tasks.flappy import agent as ag
from ptqnn.tasks.flappy import train as tr
from ptqnn.train import kwsa

ROOT = Path(__file__).resolve().parents[1]


def git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - no git available
        return "unknown"
    return out.stdout.strip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--seed", type=int, default=0, help="KW-SA / restart seed")
    p.add_argument("--iterations", type=int, default=25, help="KW-SA coordinate sweeps")
    p.add_argument("--restarts", type=int, default=300, help="random parameter vectors tried first")
    p.add_argument("--restart-steps", type=int, default=250, help="frame cap during restarts")
    p.add_argument("--max-steps", type=int, default=1500, help="frame cap per training episode")
    p.add_argument("--eval-steps", type=int, default=5000, help="frame cap when scoring")
    p.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--eval-seeds", type=int, nargs="+", default=list(range(5, 25)))
    p.add_argument("--hidden", type=int, default=4, help="prototype neurons [PTQNN §III.D]")
    p.add_argument("--gif-frames", type=int, default=300)
    p.add_argument("--out", type=Path, default=ROOT / "experiments" / "runs" / "flappy" / stamp)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    net = Network(n_inputs=2, n_hidden=args.hidden, hidden="prototype")
    kw = kwsa.KWSAConfig(
        **{**asdict(tr.DEFAULT_KWSA), "iterations": args.iterations, "seed": args.seed}
    )
    oc = tr.ObjectiveConfig(max_steps=args.max_steps)

    log: list[dict[str, Any]] = []

    def on_iteration(n: int, x: npt.NDArray[np.float64], fx: float) -> None:
        policy = tr.make_policy(net, x, oc)
        res = tr.evaluate(policy, args.eval_seeds[:1], args.eval_steps, oc.env_config)[0]
        log.append({"iteration": n, "loss": fx, "eval_score": res.score, "eval_steps": res.steps})
        print(
            f"  iter {n:3d}  loss {fx: .4f}  score {res.score:5d}  frames {res.steps:5d}",
            flush=True,
        )

    print(f"[flappy] {net.n_qubits} qubits, {net.n_params} parameters -> {out}", flush=True)
    t0 = time.perf_counter()
    result = tr.train(
        net,
        cfg=kw,
        seeds=tuple(args.train_seeds),
        max_steps=args.max_steps,
        objective_config=oc,
        n_restarts=args.restarts,
        restart_steps=args.restart_steps,
        eval_seeds=tuple(args.eval_seeds),
        eval_steps=args.eval_steps,
        callback=on_iteration,
    )
    wall = time.perf_counter() - t0

    policy = tr.make_policy(net, result.params, oc)
    episodes = tr.evaluate(
        policy, tuple(args.eval_seeds), args.eval_steps, oc.env_config, record=True
    )
    best = max(episodes, key=lambda r: (r.score, r.steps))

    np.save(out / "params.npy", result.params)
    with (out / "log.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["iteration", "loss", "eval_score", "eval_steps"])
        w.writeheader()
        w.writerows(log)

    meta = {
        "task": "flappy",
        "paper": "[PTQNN §III.D, Fig 10, 11]",
        "seed": args.seed,
        "git_hash": git_hash(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "wall_seconds": wall,
        "backend": ref.StatevectorBackend.name,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "network": {
            "n_inputs": net.n_inputs,
            "n_hidden": net.n_hidden,
            "hidden": net.hidden,
            "n_qubits": net.n_qubits,
            "n_params": net.n_params,
        },
        "kwsa": asdict(kw),
        "objective": {**asdict(oc), "env_config": asdict(oc.env_config)},
        "train_seeds": list(args.train_seeds),
        "eval_seeds": list(args.eval_seeds),
        "eval_steps": args.eval_steps,
        "n_restarts": args.restarts,
        "restart_steps": args.restart_steps,
        "init_loss": result.init_loss,
        "final_loss": result.loss,
        "loss_evaluations": result.evaluations,
        "scores": result.scores,
        "best_score": best.score,
        "best_steps": best.steps,
        "perfect": all(r.survived for r in episodes),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    gif: str | None = None
    try:
        frames = ag.render_episode(oc.env_config, best.trajectory, max_frames=args.gif_frames)
        gif = str(ag.save_gif(frames, out / "best.gif", fps=30))
    except ImportError as exc:  # pragma: no cover - optional dependency missing
        print(f"[flappy] no GIF ({exc})", flush=True)

    print(
        f"[flappy] loss {result.init_loss:.3f} -> {result.loss:.3f}; scores {result.scores}; "
        f"best {best.score} pipes in {best.steps} frames; {wall:.1f} s; {out}"
        + (f"; gif {gif}" if gif else ""),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
