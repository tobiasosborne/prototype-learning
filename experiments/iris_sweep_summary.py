"""Aggregate an Iris seed sweep into the Fig 9 verdict, both accuracies, and the Fig 7 contrast.

Reads the run directories written by ``experiments/iris_fig7_fig8_fig9.py`` (one per seed and
ambiguity policy) and writes ``experiments/runs/iris_sweep/<stamp>/{summary.json, summary.md,
meta.json}``.  Nothing is trained here.

Verdict rules (fixed before the sweep; ``[session-4 analysis]``, stated in
``ptqnn.tasks.iris.train``):

* firing (encoding-free, :func:`ptqnn.tasks.iris.train.firing_placement`): over ≥ 20 seeds the
  prototype layer's coverage and selectivity z-scores ``rho_f``, ``rho_f_selectivity`` are
  positive (median > 0) AND exceed the classic layer's on the same seeds — one-sided paired
  Wilcoxon, ``p < 0.01``, for each of the two.  Per-seed counts are reported alongside.
* placement (angle-based, :func:`ptqnn.tasks.iris.train.prototype_placement`), per angle→pattern
  encoding: median prototype ``rho ≥ 0.5`` with a bootstrap 95 % CI of the median excluding 0,
  and prototype > classic under a one-sided paired Wilcoxon test with ``p < 0.01``.

Accuracies are quoted on the policy's test set (``test``) and on every test sample
(``test_unfiltered``; review 004 M3 — the drop policy filters the test set by its own labels).
With ``--calibrated`` both are also computed with the level scale fitted once on the training
codes (``iris.model.calibrated_accuracy``) from the saved parameters: the set-refitted read-out
of the training script lets a code's label depend on which other codes are evaluated.
The Fig 7 contrast counts seeds whose final test KL exceeds its running minimum by more than
``RISE_REL`` of that minimum (``[PTQNN Fig 7]``: classic test loss rises, prototype does not).

Usage: ``uv run python experiments/iris_sweep_summary.py 'experiments/runs/iris/*-seed*-*'``.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # before numpy: one BLAS thread per pool worker
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import glob
import json
import platform
import socket
import subprocess
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("prototype", "classic")
MIN_SEEDS = 20
ALPHA = 0.01
RHO_MIN = 0.5
RISE_REL = 0.05
"""[ASSUMPTION] a test-KL rise counts when final > (1 + RISE_REL) · running minimum."""


def load_runs(dirs: list[Path]) -> list[dict[str, Any]]:
    """One record per run directory; a directory missing a file is an error, never skipped."""
    runs = []
    for d in dirs:
        rec: dict[str, Any] = {"dir": str(d)}
        for name in ("results", "meta", "losses"):
            path = Path(d) / f"{name}.json"
            if not path.is_file():
                raise FileNotFoundError(f"{path}: run incomplete ({name}.json missing)")
            rec[name] = json.loads(path.read_text())
        rec["seed"] = int(rec["meta"]["seed"])
        rec["policy"] = str(rec["meta"]["split"]["ambiguous"])
        rec["git_hash"] = str(rec["meta"]["git_hash"])
        runs.append(rec)
    return runs


def _describe(x: list[float]) -> dict[str, float]:
    a = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "median": float(np.median(a)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def _wilcoxon_greater(x: list[float], y: list[float]) -> float:
    """One-sided paired Wilcoxon p for ``x > y``; NaN when every difference is zero."""
    d = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    if np.all(d == 0):
        return float("nan")
    return float(stats.wilcoxon(d, alternative="greater").pvalue)


def _bootstrap_median_ci(x: list[float], n_boot: int, seed: int) -> tuple[float, float]:
    a = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    meds = np.median(a[rng.integers(0, a.size, size=(n_boot, a.size))], axis=1)
    lo, hi = np.quantile(meds, [0.025, 0.975])
    return float(lo), float(hi)


def _policy_summary(runs: list[dict[str, Any]], n_boot: int, boot_seed: int) -> dict[str, Any]:
    runs = sorted(runs, key=lambda r: r["seed"])
    n = len(runs)
    enough = n >= MIN_SEEDS
    out: dict[str, Any] = {"n_seeds": n, "seeds": [r["seed"] for r in runs]}

    out["accuracy"] = {
        kind: {
            key: _describe([r["results"]["accuracy"][kind][key] for r in runs])
            for key in ("test", "test_unfiltered")
        }
        for kind in KINDS
    }
    if all("calibrated" in r for r in runs):
        for kind in KINDS:
            for key in ("test", "test_unfiltered"):
                out["accuracy"][kind][f"{key}_calibrated"] = _describe(
                    [r["calibrated"][kind][key] for r in runs]
                )

    fp = [r["results"]["firing"]["prototype"] for r in runs]
    fc = [r["results"]["firing"]["classic"] for r in runs]
    firing: dict[str, Any] = {"n_draws": sorted({int(f["n_draws"]) for f in fp + fc})}
    p_vals = {}
    for key in ("rho_f", "rho_f_selectivity"):
        xp, xc = [f[key] for f in fp], [f[key] for f in fc]
        firing[key] = {"prototype": _describe(xp), "classic": _describe(xc)}
        p_vals[key] = _wilcoxon_greater(xp, xc)
    firing["wilcoxon_p"] = p_vals
    # The z-scores compare each layer with ITS OWN untrained null (what training did); the raw
    # values compare the two layers with each other.  They need not rank alike (review 005 B1:
    # untrained prototype layers already cover ~0.79, untrained classic ones ~0.39).
    firing["raw"] = {}
    for key in ("coverage", "selectivity"):
        xp, xc = [f[key] for f in fp], [f[key] for f in fc]
        firing["raw"][key] = {
            "prototype": _describe(xp),
            "classic": _describe(xc),
            "prototype_null_median": float(np.median([f.get(f"{key}_null", np.nan) for f in fp])),
            "classic_null_median": float(np.median([f.get(f"{key}_null", np.nan) for f in fc])),
            "prototype_higher": sum(a > b for a, b in zip(xp, xc, strict=True)),
        }
    firing["prototype_both_positive"] = sum(
        f["rho_f"] > 0 and f["rho_f_selectivity"] > 0 for f in fp
    )
    firing["prototype_beats_classic_both"] = sum(
        a["rho_f"] > b["rho_f"] and a["rho_f_selectivity"] > b["rho_f_selectivity"]
        for a, b in zip(fp, fc, strict=True)
    )
    firing["verdict"] = bool(
        enough
        and all(firing[k]["prototype"]["median"] > 0 for k in ("rho_f", "rho_f_selectivity"))
        and all(p < ALPHA for p in p_vals.values())
    )
    out["firing"] = firing

    placement = {}
    for enc in ("inverse", "direct"):
        rp = [r["results"]["placement"][f"prototype-{enc}"]["rho"] for r in runs]
        rc = [r["results"]["placement"][f"classic-{enc}"]["rho"] for r in runs]
        ci = _bootstrap_median_ci(rp, n_boot, boot_seed)
        p = _wilcoxon_greater(rp, rc)
        med = float(np.median(rp))
        placement[enc] = {
            "median_prototype": med,
            "median_classic": float(np.median(rc)),
            "ci95": list(ci),
            "ci95_classic": list(_bootstrap_median_ci(rc, n_boot, boot_seed)),
            "wilcoxon_p": p,
            "all_dead_prototype": sum(
                bool(r["results"]["placement"][f"prototype-{enc}"]["all_dead"]) for r in runs
            ),
            "verdict": bool(enough and med >= RHO_MIN and ci[0] > 0 and p < ALPHA),
        }
    out["placement"] = placement

    over = {}
    for kind in ("classic", "prototype"):
        rises = []
        for r in runs:
            test = np.asarray(r["losses"][kind]["test"], dtype=np.float64)
            rises.append(float(test[-1] - test.min()))
            r.setdefault("_min", {})[kind] = float(test.min())
        mins = [r["_min"][kind] for r in runs]
        over[kind] = {
            "seeds_test_kl_rises": sum(
                rise > RISE_REL * m for rise, m in zip(rises, mins, strict=True)
            ),
            "median_rise": float(np.median(rises)),
            "test_kl_final": _describe([float(r["losses"][kind]["test"][-1]) for r in runs]),
        }
    out["overfitting"] = over
    return out


def _calibrated(run_dir: str) -> dict[str, dict[str, float]]:
    """Both test accuracies with the level scale fitted on the training codes (from params)."""
    from ptqnn.tasks.iris import data as iris_data
    from ptqnn.tasks.iris import model as iris_model
    from ptqnn.tasks.iris import train as iris_train

    meta = json.loads((Path(run_dir) / "meta.json").read_text())
    sp = meta["split"]
    split = iris_data.load_split(
        seed=int(meta["seed"]), train_size=float(sp["train_size"]), ambiguous=sp["ambiguous"]
    )
    fields = iris_train.IrisConfig.__dataclass_fields__
    cfg = iris_train.IrisConfig(**{k: v for k, v in meta["config"].items() if k in fields})
    out = {}
    for kind in KINDS:
        net = iris_train.network(cfg, kind)
        p = np.load(Path(run_dir) / f"params-{kind}.npy")
        acc = iris_model.calibrated_accuracy
        out[kind] = {
            "test": acc(net, p, split.train_codes, split.test_codes, split.test_labels),
            "test_unfiltered": acc(
                net, p, split.train_codes, split.test_codes_all, split.test_labels_all
            ),
        }
    return out


def resolve_dirs(patterns: list[str]) -> list[str]:
    """Run dirs matched by the globs; a pattern that matches nothing, or no match at all, raises."""
    if not patterns:
        raise ValueError("no run-dir patterns given")
    dirs: set[str] = set()
    for pat in patterns:
        hit = [p for p in glob.glob(pat) if Path(p).is_dir()]
        if not hit:
            raise ValueError(f"pattern {pat!r} matches no run directory")
        dirs.update(hit)
    return sorted(dirs)


def aggregate(
    runs: list[dict[str, Any]], n_boot: int = 10_000, boot_seed: int = 0
) -> dict[str, Any]:
    """Per-policy summary; refuses duplicate (policy, seed) pairs and runs from mixed commits."""
    hashes = {r["git_hash"] for r in runs}
    if len(hashes) > 1:
        raise ValueError(f"runs come from different git commits: {sorted(hashes)}")
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        if any(o["seed"] == r["seed"] for o in by_policy[r["policy"]]):
            raise ValueError(f"policy {r['policy']}: seed {r['seed']} appears twice")
        by_policy[r["policy"]].append(r)
    return {pol: _policy_summary(rs, n_boot, boot_seed) for pol, rs in sorted(by_policy.items())}


def _fmt(d: dict[str, float], pct: bool = False) -> str:
    s = 100.0 if pct else 1.0
    return (
        f"{s * d['mean']:.1f} ± {s * d['std']:.1f} (min {s * d['min']:.1f})"
        if pct
        else (f"median {d['median']:+.2f} (range {d['min']:+.2f} … {d['max']:+.2f})")
    )


def markdown(summary: dict[str, Any], dirs: list[str], git_hash: str) -> str:
    lines = [
        "# Iris seed sweep — Fig 9 verdict, accuracies, Fig 7 contrast",
        "",
        f"{len(dirs)} run dirs, training commit `{git_hash[:12]}`.  Rules: module docstring of "
        "`experiments/iris_sweep_summary.py`.",
        "",
    ]
    for pol, s in summary.items():
        f, acc = s["firing"], s["accuracy"]
        lines += [
            f"## Policy `{pol}` — {s['n_seeds']} seeds",
            "",
            "| | prototype | classic |",
            "|---|---|---|",
            f"| test acc. (policy set) % | {_fmt(acc['prototype']['test'], True)} | "
            f"{_fmt(acc['classic']['test'], True)} |",
            f"| test acc. (all samples) % | {_fmt(acc['prototype']['test_unfiltered'], True)} | "
            f"{_fmt(acc['classic']['test_unfiltered'], True)} |",
            *(
                [
                    f"| test acc., train-fitted scale (policy set) % | "
                    f"{_fmt(acc['prototype']['test_calibrated'], True)} | "
                    f"{_fmt(acc['classic']['test_calibrated'], True)} |",
                    f"| test acc., train-fitted scale (all samples) % | "
                    f"{_fmt(acc['prototype']['test_unfiltered_calibrated'], True)} | "
                    f"{_fmt(acc['classic']['test_unfiltered_calibrated'], True)} |",
                ]
                if "test_calibrated" in acc["prototype"]
                else []
            ),
            f"| firing coverage (raw; untrained median) | "
            f"{f['raw']['coverage']['prototype']['median']:.3f} "
            f"({f['raw']['coverage']['prototype_null_median']:.3f}) | "
            f"{f['raw']['coverage']['classic']['median']:.3f} "
            f"({f['raw']['coverage']['classic_null_median']:.3f}) |",
            f"| firing selectivity (raw; untrained median) | "
            f"{f['raw']['selectivity']['prototype']['median']:.3f} "
            f"({f['raw']['selectivity']['prototype_null_median']:.3f}) | "
            f"{f['raw']['selectivity']['classic']['median']:.3f} "
            f"({f['raw']['selectivity']['classic_null_median']:.3f}) |",
            f"| firing coverage z | {_fmt(f['rho_f']['prototype'])} | "
            f"{_fmt(f['rho_f']['classic'])} |",
            f"| firing selectivity z | {_fmt(f['rho_f_selectivity']['prototype'])} | "
            f"{_fmt(f['rho_f_selectivity']['classic'])} |",
            f"| seeds whose test KL rises | {s['overfitting']['prototype']['seeds_test_kl_rises']}"
            f" | {s['overfitting']['classic']['seeds_test_kl_rises']} |",
            "",
            f"Raw firing: prototype covers more on {f['raw']['coverage']['prototype_higher']}/"
            f"{s['n_seeds']} seeds, is more selective on "
            f"{f['raw']['selectivity']['prototype_higher']}/{s['n_seeds']}.  z-scores (below) are "
            "against each layer's own untrained null, i.e. what training changed.",
            "",
            f"Firing: prototype both z > 0 on {f['prototype_both_positive']}/{s['n_seeds']} "
            f"seeds, beats classic on both on {f['prototype_beats_classic_both']}/{s['n_seeds']}; "
            f"one-sided Wilcoxon p (prototype > classic): coverage {f['wilcoxon_p']['rho_f']:.2g}, "
            f"selectivity {f['wilcoxon_p']['rho_f_selectivity']:.2g}; null draws {f['n_draws']}. "
            f"**Verdict: {'reproduced' if f['verdict'] else 'not reproduced'}.**",
            "",
        ]
        for enc, p in s["placement"].items():
            lines.append(
                f"Placement ({enc}): median ρ prototype {p['median_prototype']:+.2f} "
                f"(95 % CI {p['ci95'][0]:+.2f} … {p['ci95'][1]:+.2f}) vs classic "
                f"{p['median_classic']:+.2f} (CI {p['ci95_classic'][0]:+.2f} … "
                f"{p['ci95_classic'][1]:+.2f}); Wilcoxon p {p['wilcoxon_p']:.2g}. "
                f"**Verdict: {'reproduced' if p['verdict'] else 'not reproduced'}.**"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _git_hash() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT)
    return r.stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("patterns", nargs="+", help="glob(s) of run dirs")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--calibrated", action="store_true", help="also score with the train-fitted level scale"
    )
    ap.add_argument("--processes", type=int, default=32)
    args = ap.parse_args()
    t0 = time.perf_counter()
    dirs = resolve_dirs(args.patterns)
    runs = load_runs([Path(d) for d in dirs])
    if args.calibrated:
        from multiprocessing import Pool

        with Pool(min(args.processes, len(runs))) as pool:
            for r, c in zip(runs, pool.map(_calibrated, [r["dir"] for r in runs]), strict=True):
                r["calibrated"] = c
    summary = aggregate(runs, n_boot=args.n_boot)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or ROOT / "experiments" / "runs" / "iris_sweep" / stamp
    out.mkdir(parents=True, exist_ok=True)
    train_hash = runs[0]["git_hash"] if runs else ""
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "summary.md").write_text(markdown(summary, dirs, train_hash))
    meta = {
        "experiment": "iris_sweep_summary",
        "timestamp": stamp,
        "git_hash": _git_hash(),
        "training_git_hash": train_hash,
        "run_dirs": [str(Path(d).resolve().relative_to(ROOT)) for d in dirs],
        "n_boot": args.n_boot,
        "calibrated": args.calibrated,
        "boot_seed": 0,
        "rules": {"min_seeds": MIN_SEEDS, "alpha": ALPHA, "rho_min": RHO_MIN, "rise_rel": RISE_REL},
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "wall_seconds": time.perf_counter() - t0,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print((out / "summary.md").read_text())
    print(f"→ {out}")


if __name__ == "__main__":
    main()
