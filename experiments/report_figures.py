"""Every figure of the LaTeX report (``reports/tex/``), regenerated from the run directories.

One style, one palette, one script: each function below reads a run dir's ``results.json`` /
``.npz`` / ``.npy`` (never a PNG), and writes ``reports/tex/figures/<name>.pdf`` + ``.png``.
Run dirs are pinned in :data:`RUNS`; a missing file is an error, not a skipped figure.

    LD_LIBRARY_PATH=/usr/lib/wsl/lib uv run python experiments/report_figures.py   # ≈ 1 min

Palette (fixed slot order, never cycled; categorical identity only): prototype = blue,
classic = orange, paper = grey; devices keep one hue each across every figure.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from ptqnn.backends import reference as ref
from ptqnn.backends.shots import ShotBackend
from ptqnn.circuits.neurons import Network
from ptqnn.tasks import bitcount
from ptqnn.tasks.ecg import data as ecg_data
from ptqnn.tasks.ecg import model as ecg_model
from ptqnn.tasks.flappy import agent as flappy_agent
from ptqnn.tasks.iris import data as iris_data
from ptqnn.tasks.iris import model as iris_model
from ptqnn.tasks.iris import train as iris_train

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/tex/figures"
RUNS = {
    "bitcount": ROOT / "experiments/runs/bitcount/20260820T135930Z",
    "bitcount_noise": ROOT / "experiments/runs/bitcount_noise/20260820T145335Z",
    "bitcount_shots": ROOT / "experiments/runs/bitcount_shots/20260821T074944Z",
    "flappy": ROOT / "experiments/runs/flappy/20260820T143753Z",
    "flappy_noise": ROOT / "experiments/runs/flappy_noise/20260820T154249Z",
    "flappy_shots": ROOT / "experiments/runs/flappy_shots/20260821T075321Z",
    "ecg": ROOT / "experiments/runs/ecg/20260820T162014",
    "ecg_physionet": ROOT / "experiments/runs/ecg/20260820T165912",
    "iris": [
        ROOT / "experiments/runs/iris/20260820T144849Z",
        ROOT / "experiments/runs/iris/20260820T145454Z",
        ROOT / "experiments/runs/iris/20260820T145455Z",
    ],
    "ecg_iris_noise": ROOT / "experiments/runs/ecg_iris_noise/20260820T155618Z",
    "hardware_plan": ROOT / "experiments/runs/hardware-plan/20260821T073211Z",
}

# --- style --------------------------------------------------------------------------------------

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)
PROTO, CLASSIC, PAPER = BLUE, ORANGE, "#8a8a86"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
DEVICE_ORDER = [
    "quantinuum_h2_1",
    "quantinuum_h1_1",
    "ionq_forte",
    "google_willow",
    "ibm_fez",
    "iqm_garnet",
    "rigetti_ankaa_3",
]
DEVICE_LABEL = {
    "quantinuum_h2_1": "Quantinuum H2-1",
    "quantinuum_h1_1": "Quantinuum H1-1",
    "ionq_forte": "IonQ Forte",
    "google_willow": "Google Willow",
    "ibm_fez": "IBM Fez",
    "iqm_garnet": "IQM Garnet",
    "rigetti_ankaa_3": "Rigetti Ankaa-3",
}
DEVICE_COLOR = dict(
    zip(DEVICE_ORDER, [BLUE, AQUA, GREEN, VIOLET, ORANGE, YELLOW, RED], strict=True)
)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "pdf.fonttype": 42,
    }
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    log(f"wrote {name}")


def load(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def pct(ax: plt.Axes, axis: str = "y") -> None:
    fmt = FuncFormatter(lambda v, _: f"{100 * v:.0f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


# --- 1. reproductions ----------------------------------------------------------------------------


def fig_activation_surface() -> None:
    """Fig 3 of the paper: activation P(out=1) of one prototype neuron over the two inputs."""
    grid = np.load(RUNS["bitcount"] / "fig03_grid.npy")
    n = grid.shape[0]
    fig, ax = plt.subplots(figsize=(3.6, 3.1))
    im = ax.imshow(grid.T, origin="lower", extent=(0, 1, 0, 1), cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("input 1 (probability of |1⟩)")
    ax.set_ylabel("input 2 (probability of |1⟩)")
    ax.grid(False)
    for (x, y), t in (((0, 1), "1.00"), ((1, 0), "0.00"), ((0, 0), "0.50"), ((1, 1), "0.50")):
        ax.annotate(
            t,
            (x, y),
            xytext=(6 if x == 0 else -6, 6 if y == 0 else -6),
            textcoords="offset points",
            ha="left" if x == 0 else "right",
            va="bottom" if y == 0 else "top",
            fontsize=8,
            color=INK,
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85},
        )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("P(out = 1)")
    ax.set_title(f"Prototype neuron for the pattern (0, 1), {n}×{n} grid")
    save(fig, "fig_activation_surface")


def fig_bitcount_histograms() -> None:
    """Fig 5: target histogram (popcount+1)/112 against the trained model's P(x | out = 1)."""
    r = load(RUNS["bitcount_noise"] / "results.json")
    target = np.asarray(r["target"])
    model = np.asarray(r["A"]["prototype"]["clean_distribution"])
    x = np.arange(32)
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ax.bar(x - 0.2, target, width=0.4, color=PAPER, label="target (number of ones + 1)/112")
    ax.bar(
        x + 0.2,
        model,
        width=0.4,
        color=PROTO,
        label=f"trained prototype network, KL = {r['A']['prototype']['clean_kl']:.1e}",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([format(i, "05b") for i in x], rotation=90, fontsize=5.5, family="monospace")
    ax.set_xlabel("input bit string (qubit 1 first)")
    ax.set_ylabel("probability")
    ax.legend(loc="upper left")
    save(fig, "fig_bitcount_histograms")


def fig_bitcount_convergence() -> None:
    """Fig 6: KL vs training sweep, superposition vs our iterative baseline, 6 seeds."""
    h = load(RUNS["bitcount"] / "histories.json")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
    for ax, key, color, title in (
        (axes[0], "superposition", PROTO, "superposition training"),
        (axes[1], "iterative", ORANGE, "iterative (one input per step) — our construction"),
    ):
        for s, hist in h[key].items():
            ax.plot(hist, color=color, alpha=0.55, lw=1.1)
        med = np.median(np.array(list(h[key].values())), axis=0)
        ax.plot(med, color=INK, lw=1.8, label="median of 6 seeds")
        ax.set_yscale("log")
        ax.set_xlabel("sweep")
        ax.set_title(title)
        ax.axhline(5e-3, color=GRID, lw=1)
    axes[0].set_ylabel("KL(target ‖ model)")
    fig.text(
        0.5,
        -0.02,
        "one sweep = 3 × 36 circuit evaluations (6 optimiser seeds per panel)",
        ha="center",
        fontsize=7.5,
        color=INK2,
    )
    axes[0].annotate(
        "paper: ≈0 after ~8 iterations",
        (9, 5e-3),
        xytext=(10.5, 2.5e-2),
        fontsize=7.5,
        color=INK2,
        arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.7},
    )
    axes[1].annotate(
        "paper: ≈0 after ~120 iterations; ours stalls ≈0.15",
        (120, 0.15),
        xytext=(60, 0.6),
        fontsize=7.5,
        color=INK2,
        arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.7},
    )
    axes[0].legend(loc="lower left")
    save(fig, "fig_bitcount_convergence")


def fig_iris_losses() -> None:
    """Fig 7: train / test KL per sweep for the classic and the prototype network (3 splits)."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
    for ax, kind, color, title in (
        (axes[0], "classic", CLASSIC, "classic hidden neurons"),
        (axes[1], "prototype", PROTO, "prototype hidden neurons"),
    ):
        for i, run in enumerate(RUNS["iris"]):
            L = load(run / "losses.json")[kind]
            ax.plot(L["train"], color=color, lw=1.2, alpha=0.9, label="train" if i == 0 else None)
            ax.plot(
                L["test"], color=color, lw=1.2, ls="--", alpha=0.9, label="test" if i == 0 else None
            )
        ax.set_yscale("log")
        ax.set_xlabel("sweep")
        ax.set_title(title)
        ax.legend(loc="upper right")
    axes[0].set_ylabel("KL(target ‖ model), 3 data splits")
    save(fig, "fig_iris_losses")


def fig_iris_histograms() -> None:
    """Fig 8: predicted vs target weight per distinct 12-bit code, test split, seed 0."""
    run = RUNS["iris"][0]
    meta = load(run / "meta.json")
    split = iris_data.load_split(seed=meta["seed"], train_size=meta["split"]["train_size"])
    cfg = iris_train.IrisConfig(**meta["config"])
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.4), sharex=True)
    for ax, kind, color in ((axes[0], "classic", CLASSIC), (axes[1], "prototype", PROTO)):
        net = iris_train.network(cfg, kind)
        params = np.load(run / f"params-{kind}.npy")
        uniq, values = iris_model.regressed_values(net, params, split.test_codes)
        target = iris_model.target_distribution(split.test_codes, split.test_labels)[uniq]
        x = np.arange(uniq.size)
        ax.bar(x - 0.2, target, width=0.4, color=PAPER, label="target (weight ∝ (label+1)²)")
        ax.bar(x + 0.2, values, width=0.4, color=color, label=f"{kind} network")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [iris_data.format_code(int(c)) for c in uniq],
            rotation=90,
            fontsize=5.5,
            family="monospace",
        )
        ax.set_ylim(0, 0.105)
        ax.set_ylabel("probability")
        ax.legend(loc="upper right", ncol=2)
        ax.set_title(f"{kind}: test split, {uniq.size} distinct codes", loc="left")
    axes[1].set_xlabel("12-bit code (feature 1 clusters | feature 2 | feature 3 | feature 4)")
    save(fig, "fig_iris_histograms")


def fig_flappy_policy() -> None:
    """The trained Flappy policy: P(out=1) over the two observations and the flap boundary."""
    meta = load(RUNS["flappy"] / "meta.json")
    params = np.load(RUNS["flappy"] / "params.npy")
    net = Network(n_inputs=2, n_hidden=meta["network"]["n_hidden"], hidden="prototype")
    pol = flappy_agent.QuantumFlappyPolicy(net=net, params=params, cache_levels=0)
    levels = 65
    xs = np.linspace(0, 1, levels)
    P = np.array([[pol.activation(np.array([a, b])) for b in xs] for a in xs])
    fig, ax = plt.subplots(figsize=(3.9, 3.2))
    im = ax.imshow(P.T, origin="lower", extent=(0, 1, 0, 1), cmap="RdBu_r", vmin=0, vmax=1)
    ax.contour(xs, xs, P.T, levels=[0.5], colors=[INK], linewidths=1.2)
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("bird height (normalised)")
    ax.set_ylabel("next gap centre (normalised)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("P(out = 1); flap above ½ (black line)")
    ax.set_title("Trained 7-qubit Flappy policy")
    save(fig, "fig_flappy_policy")


def fig_ecg_trace(run_key: str, name: str, title: str) -> None:
    """Fig 13: clean, noisy and denoised ECG for the 3-prototype network."""
    z = np.load(RUNS[run_key] / "params.npz")
    clean, noisy = z["clean"], z["noisy"]
    res = load(RUNS[run_key] / "results.json")
    X, _ = ecg_data.windows(noisy)
    fig, axes = plt.subplots(
        2, 1, figsize=(6.6, 3.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]}
    )
    t = np.arange(clean.size)
    axes[0].plot(t, noisy, color=PAPER, lw=0.6, label=f"noisy input ({res['input_psnr']:.1f} dB)")
    axes[0].plot(t, clean, color=INK, lw=1.0, label="clean signal")
    for kind, color, k in (("prototype", PROTO, 3), ("classic", CLASSIC, 3)):
        row = next(r for r in res["rows"] if r["kind"] == kind and r["n_hidden"] == k)
        net = Network(n_inputs=5, n_hidden=k, hidden=kind)
        yhat = ecg_model.predict(net, z[f"{kind}_{k}"], X, encoding=row["encoding"])
        axes[0].plot(
            t, yhat, color=color, lw=1.0, label=f"{kind}, {k} hidden ({row['psnr']:.1f} dB)"
        )
        axes[1].plot(t, yhat - clean, color=color, lw=0.7)
    axes[1].plot(t, noisy - clean, color=PAPER, lw=0.5, zorder=0)
    axes[0].set_ylabel("amplitude (scaled to [0, 1])")
    axes[1].set_ylabel("error")
    axes[1].set_xlabel("sample")
    axes[0].legend(ncol=2, loc="upper right")
    axes[0].set_title(title, loc="left")
    save(fig, name)


def fig_ecg_table() -> None:
    """Table I as a chart: PSNR per model and hidden count, paper vs ours (synthetic signal)."""
    res = load(RUNS["ecg"] / "results.json")
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    xs = np.arange(3)
    w = 0.2
    for j, (kind, color) in enumerate((("classic", CLASSIC), ("prototype", PROTO))):
        rows = sorted((r for r in res["rows"] if r["kind"] == kind), key=lambda r: r["n_hidden"])
        ax.bar(
            xs + (j - 0.5) * 2 * w - w / 2,
            [r["paper_psnr"] for r in rows],
            width=w,
            color=color,
            alpha=0.35,
            label=f"{kind}, paper",
        )
        ax.bar(
            xs + (j - 0.5) * 2 * w + w / 2,
            [r["psnr"] for r in rows],
            width=w,
            color=color,
            label=f"{kind}, ours",
        )
        for x0, r in zip(xs, rows, strict=True):
            ax.text(
                x0 + (j - 0.5) * 2 * w + w / 2,
                r["psnr"] + 0.3,
                f"{r['psnr']:.1f}",
                ha="center",
                fontsize=6.5,
                color=INK,
            )
    ax.axhline(
        res["input_psnr"],
        color=PAPER,
        ls="--",
        lw=1,
        label=f"noisy input {res['input_psnr']:.1f} dB",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(["1 hidden", "2 hidden", "3 hidden"])
    ax.set_ylabel("PSNR (dB), test half")
    ax.set_ylim(30, 42)
    ax.legend(ncol=2, loc="upper left")
    save(fig, "fig_ecg_table")


# --- 2. noise ------------------------------------------------------------------------------------


def fig_bitcount_noise_devices() -> None:
    r = load(RUNS["bitcount_noise"] / "results.json")
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    for ax, key, ylabel in (
        (axes[0], "kl_target", "KL(target ‖ noisy model)"),
        (axes[1], "tv_clean_vs_noisy", "total variation, clean vs noisy output"),
    ):
        ys = np.arange(len(DEVICE_ORDER))
        for j, (kind, color) in enumerate((("prototype", PROTO), ("classic", CLASSIC))):
            vals = [r["A"][kind]["per_device"][d][key] for d in DEVICE_ORDER]
            ax.barh(ys + (j - 0.5) * 0.38, vals, height=0.36, color=color, label=kind)
        if key == "kl_target":
            for kind, color in (("prototype", PROTO), ("classic", CLASSIC)):
                ax.axvline(r["A"][kind]["clean_kl"], color=color, ls=":", lw=1)
            ax.text(
                r["A"]["prototype"]["clean_kl"],
                len(DEVICE_ORDER) - 0.4,
                " noiseless",
                fontsize=6.5,
                color=INK2,
            )
        ax.set_yticks(ys)
        ax.set_yticklabels([DEVICE_LABEL[d] for d in DEVICE_ORDER])
        ax.invert_yaxis()
        ax.set_xlabel(ylabel)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.suptitle(
        "Trained bit-count networks (11 qubits) evaluated under each vendor's calibrated noise",
        fontsize=9,
    )
    save(fig, "fig_bitcount_noise_devices")


def fig_bitcount_noise_scale() -> None:
    r = load(RUNS["bitcount_noise"] / "results.json")["C"]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for kind, color in (("prototype", PROTO), ("classic", CLASSIC)):
        s = sorted(float(k) for k in r["fig5"][kind])
        kl = [r["fig5"][kind][str(int(v)) if v == int(v) else str(v)]["kl_target"] for v in s]
        ax.plot(s, kl, marker="o", color=color, label=f"{kind}, evaluated at scale s")
        ret = r["retrained"][kind]
        s2 = sorted(float(k) for k in ret)
        kl2 = [ret[str(int(v)) if v == int(v) else str(v)]["noisy_kl"] for v in s2]
        ax.plot(
            s2,
            kl2,
            marker="s",
            ls="--",
            color=color,
            alpha=0.8,
            label=f"{kind}, retrained under s (7-qubit net)",
        )
    ax.set_xlabel("noise scale s (IBM Fez calibration × s)")
    ax.set_ylabel("KL(target ‖ model)")
    ax.set_yscale("log")
    ax.set_xscale("symlog", linthresh=0.25)
    ax.legend(loc="upper left")
    ax.set_title(
        f"Response to noise strength; fitted KL ∝ s$^{{{r['slope_prototype']['loglog_exponent']:.1f}}}$",
        loc="left",
    )
    save(fig, "fig_bitcount_noise_scale")


def fig_bitcount_noise_aware() -> None:
    r = load(RUNS["bitcount_noise"] / "results.json")["B"]
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    ys = np.arange(len(DEVICE_ORDER))
    for j, (kind, color) in enumerate((("prototype", PROTO), ("classic", CLASSIC))):
        before = [r["baseline"][kind][d]["kl_target"] for d in DEVICE_ORDER]
        after = [r["per_device"][kind][d]["median_noisy_kl"] for d in DEVICE_ORDER]
        ax.scatter(
            before,
            ys + (j - 0.5) * 0.3,
            color=color,
            marker="o",
            s=28,
            label=f"{kind}: trained noiselessly, evaluated noisy",
        )
        ax.scatter(
            after,
            ys + (j - 0.5) * 0.3,
            color=color,
            marker="D",
            s=26,
            facecolors="white",
            linewidths=1.4,
            label=f"{kind}: trained under the same noise",
        )
        for b, a, y in zip(before, after, ys + (j - 0.5) * 0.3, strict=True):
            ax.annotate(
                "",
                (a, y),
                (b, y),
                arrowprops={
                    "arrowstyle": "->",
                    "color": color,
                    "lw": 0.9,
                    "shrinkA": 3,
                    "shrinkB": 3,
                },
            )
    ax.set_yticks(ys)
    ax.set_yticklabels([DEVICE_LABEL[d] for d in DEVICE_ORDER])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel(f"KL(target ‖ model) under the device's noise — {r['n_qubits']}-qubit network")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=6.5)
    save(fig, "fig_bitcount_noise_aware")


def fig_bitcount_plugin_bias() -> None:
    r = load(RUNS["bitcount_noise"] / "results.json")["D"]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for kind, color in (("prototype", PROTO), ("classic", CLASSIC)):
        rows = [x for x in r[kind]["clean"] if x["shots"] > 0]
        n = [x["shots"] for x in rows]
        ax.errorbar(
            n,
            [x["mean"] for x in rows],
            yerr=[x["sd"] for x in rows],
            marker="o",
            color=color,
            capsize=3,
            label=f"{kind}: plug-in KL from N shots",
        )
        ax.axhline(rows[0]["exact"], color=color, ls=":", lw=1)
    nn = np.logspace(2.8, 4.3, 20)
    ax.plot(nn, 31 / (2 * 0.487 * nn), color=PAPER, ls="--", label="bias (K−1)/(2·P(out=1)·N)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("shots N")
    ax.set_ylabel("estimated KL (mean ± sd over 20 draws)")
    ax.legend(loc="upper right", fontsize=6.5)
    save(fig, "fig_bitcount_plugin_bias")


def fig_flappy_noise() -> None:
    r = load(RUNS["flappy_noise"] / "results.json")
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    ys = np.arange(len(DEVICE_ORDER))
    ax = axes[0]
    ax.barh(
        ys,
        [r["A"]["per_device"][d]["flip_fraction_frames"] for d in DEVICE_ORDER],
        color=[DEVICE_COLOR[d] for d in DEVICE_ORDER],
        height=0.6,
    )
    for y, d in zip(ys, DEVICE_ORDER, strict=True):
        row = r["A"]["per_device"][d]
        ax.text(
            row["flip_fraction_frames"] + 0.001,
            y,
            f"{row['score']['perfect_fraction']:.0%} perfect",
            va="center",
            fontsize=6.5,
            color=INK2,
        )
    ax.set_yticks(ys)
    ax.set_yticklabels([DEVICE_LABEL[d] for d in DEVICE_ORDER])
    ax.invert_yaxis()
    pct(ax, "x")
    ax.set_xlim(0, 0.07)
    ax.set_xlabel("played frames whose decision changes under the device's noise")
    ax = axes[1]
    rows = r["B"]["rows"]
    s = [x["scale"] for x in rows]
    ax.plot(
        s,
        [x["score"]["perfect_fraction"] for x in rows],
        marker="o",
        color=ORANGE,
        label="perfect games",
    )
    ax.plot(
        s,
        [x["flip_fraction_frames"] for x in rows],
        marker="s",
        color=INK2,
        label="frames decided differently",
    )
    pct(ax)
    ax.set_xlabel("noise scale s (IBM Fez × s)")
    ax.legend(loc="center right")
    ax.set_title("Noise-strength sweep, 20 held-out games", loc="left")
    save(fig, "fig_flappy_noise")


def fig_flappy_shot_noise() -> None:
    r = load(RUNS["flappy_noise"] / "results.json")["C"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    ax = axes[0]
    for cond, color, label in (
        ("clean", INK, "noiseless simulator"),
        ("quantinuum_h2_1", DEVICE_COLOR["quantinuum_h2_1"], "Quantinuum H2-1"),
        ("ibm_fez", DEVICE_COLOR["ibm_fez"], "IBM Fez"),
    ):
        rows = [x for x in r["per_condition"][cond] if x["shots"] > 0]
        ax.plot(
            [x["shots"] for x in rows],
            [x["score"]["perfect_fraction"] for x in rows],
            marker="o",
            color=color,
            label=label,
        )
    ax.set_xscale("log")
    pct(ax)
    ax.set_xlabel("shots per frame N")
    ax.set_ylabel("perfect games (20 held-out seeds)")
    ax.legend(loc="upper left")
    ax = axes[1]
    m = r["margins"]
    edges, counts = np.asarray(m["bin_edges"]), np.asarray(m["counts"])
    ax.bar(
        edges[:-1],
        counts / counts.sum(),
        width=np.diff(edges),
        align="edge",
        color=BLUE,
        alpha=0.85,
    )
    for n, ls in ((1000, ":"), (4000, "--"), (10000, "-.")):
        ax.axvline(1 / (2 * math.sqrt(n)), color=INK2, ls=ls, lw=1, label=f"δ = 1/(2√N), N = {n:,}")
    ax.set_xlim(0, 0.14)
    ax.set_xlabel("decision margin δ = |P(out=1) − ½| of played frames")
    ax.set_ylabel("fraction of frames")
    ax.legend(loc="upper right", fontsize=6.5)
    ax.set_title(f"median δ = {m['quantiles'][3]:.3f}, 100 000 frames", loc="left")
    save(fig, "fig_flappy_shot_noise")


def fig_ecg_iris_noise() -> None:
    r = load(RUNS["ecg_iris_noise"] / "results.json")
    A = r["A"]
    keys = [c["key"] for c in A["cells"]]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    ax = axes[0]
    labels = [f"{'C' if c['kind'] == 'classic' else 'P'}{c['n_hidden']}" for c in A["cells"]]
    xs = np.arange(len(keys))
    ax.bar(
        xs - 0.27, [A["clean"][k]["psnr"] for k in keys], width=0.26, color=PAPER, label="noiseless"
    )
    for j, d in enumerate(("quantinuum_h2_1", "ibm_fez")):
        vals = [A["per_device"][k][d]["psnr"] for k in keys]
        ax.bar(xs + j * 0.27, vals, width=0.26, color=DEVICE_COLOR[d], label=DEVICE_LABEL[d])
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("C = classic, P = prototype; digit = hidden neurons")
    ax.set_ylim(30, 45)
    ax.set_ylabel("PSNR (dB) on an 8-window block")
    ax.legend(loc="upper left", fontsize=6.5)
    ax.set_title("ECG — reduced run (8 windows)", loc="left")
    ax = axes[1]
    b = r["B"]
    kinds = ("classic", "prototype")
    xs = np.arange(2)
    ax.bar(
        xs - 0.27,
        [b["clean"][k]["accuracy"] for k in kinds],
        width=0.26,
        color=PAPER,
        label="noiseless",
    )
    for j, d in enumerate(("quantinuum_h2_1", "ibm_fez")):
        vals = [b["per_device"][k][d]["accuracy_mean"] for k in kinds]
        err = [b["per_device"][k][d]["accuracy_sd"] for k in kinds]
        ax.bar(
            xs + j * 0.27,
            vals,
            width=0.26,
            color=DEVICE_COLOR[d],
            yerr=err,
            capsize=3,
            label=DEVICE_LABEL[d],
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(kinds)
    pct(ax)
    ax.set_ylabel("Iris test accuracy")
    ax.set_title(f"Iris — reduced run ({b['trajectories']} trajectories)", loc="left")
    ax.legend(loc="lower left", fontsize=6.5)
    save(fig, "fig_ecg_iris_noise")


# --- 3. hardware ---------------------------------------------------------------------------------


def fig_hardware_entanglers() -> None:
    num = load(RUNS["hardware_plan"] / "numbers.json")
    circuits = ["flappy_7q", "ecg_9q", "bitcount_11q", "iris_17q"]
    labels = ["Flappy (7 q)", "ECG, 3 hidden (9 q)", "bit-count (11 q)", "Iris, 4 hidden (17 q)"]
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    xs = np.arange(len(circuits))
    w = 0.11
    for j, d in enumerate(DEVICE_ORDER):
        n2 = [num["compiled"][c][d].get("n_2q", np.nan) for c in circuits]
        sw = [3 * num["compiled"][c][d].get("n_swap", 0) for c in circuits]
        off = (j - 3) * w
        ax.bar(xs + off, np.subtract(n2, sw), width=w, color=DEVICE_COLOR[d], label=DEVICE_LABEL[d])
        ax.bar(
            xs + off,
            sw,
            width=w,
            bottom=np.subtract(n2, sw),
            color=DEVICE_COLOR[d],
            alpha=0.4,
            hatch="///",
            edgecolor="white",
            lw=0,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_ylim(10, 4000)
    ax.set_ylabel("native two-qubit gates after compilation")
    ax.legend(
        ncol=7,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        fontsize=6.5,
        handlelength=1.2,
        columnspacing=0.9,
    )
    ax.set_title(
        "Solid: gates of the network; hatched: gates spent on routing (3 per SWAP)", loc="left"
    )
    save(fig, "fig_hardware_entanglers")


def fig_hardware_fidelity() -> None:
    num = load(RUNS["hardware_plan"] / "numbers.json")
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), sharey=True)
    for ax, c, title in (
        (axes[0], "flappy_7q", "Flappy, 7 qubits"),
        (axes[1], "bitcount_11q", "bit-count, 11 qubits"),
    ):
        ys = np.arange(len(DEVICE_ORDER))
        parts = [
            ("F_2q", "two-qubit gates", BLUE),
            ("F_1q", "one-qubit gates", AQUA),
            ("F_idle", "idling", YELLOW),
            ("F_readout", "readout", VIOLET),
        ]
        left = np.zeros(len(DEVICE_ORDER))
        for key, label, color in parts:
            loss = np.array(
                [
                    -(math.log(num["fidelity"][c][d][key]))
                    if num["fidelity"][c][d].get(key)
                    else 0.0
                    for d in DEVICE_ORDER
                ]
            )
            ax.barh(ys, loss, left=left, height=0.6, color=color, label=label)
            left += loss
        for y, d in zip(ys, DEVICE_ORDER, strict=True):
            f = num["fidelity"][c][d]
            tvd = f.get("sim_tvd")
            ax.text(
                left[y] + 0.05,
                y,
                f"F ≈ {f['F_total']:.2f}"
                + (f", TVD {tvd:.2f}" if tvd is not None else "")
                + (" (bound)" if f["upper_bound_only"] else ""),
                va="center",
                fontsize=6.5,
                color=INK2,
            )
        ax.set_yticks(ys)
        ax.set_yticklabels([DEVICE_LABEL[d] for d in DEVICE_ORDER])
        ax.invert_yaxis()
        ax.set_xlabel("−log(fidelity) — stacked error budget")
        ax.set_xlim(0, max(6.5, left.max() * 1.5))
        ax.set_title(title, loc="left")
    axes[0].legend(loc="lower right", fontsize=6.5)
    save(fig, "fig_hardware_fidelity")


def fig_hardware_qpu_time() -> None:
    num = load(RUNS["hardware_plan"] / "numbers.json")
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    ys = np.arange(len(DEVICE_ORDER))
    for j, (c, shots, color, label) in enumerate(
        (
            ("flappy_7q", 4000, BLUE, "Flappy, 4 000 shots"),
            ("bitcount_11q", 64000, ORANGE, "bit-count, 64 000 shots"),
        )
    ):
        t = []
        for d in DEVICE_ORDER:
            ns = num["compiled"][c][d].get("critical_path_ns")
            t.append(shots * ns * 1e-9 if ns else np.nan)
        ax.barh(ys + (j - 0.5) * 0.36, t, height=0.34, color=color, label=label)
        for y, v, d in zip(ys, t, DEVICE_ORDER, strict=True):
            if np.isnan(v) and j == 0:
                ax.text(
                    0.012, y, "no gate durations published", va="center", fontsize=6.5, color=INK2
                )
    ax.set_yticks(ys)
    ax.set_yticklabels([DEVICE_LABEL[d] for d in DEVICE_ORDER])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(0.01, 5000)
    ax.set_xlabel("circuit time per job (s), excluding queue and per-job overhead")
    ax.legend(loc="lower right", fontsize=6.5)
    save(fig, "fig_hardware_qpu_time")


# --- 4. shot-noise programme ---------------------------------------------------------------------


def fig_bitcount_sampled_training() -> None:
    r = load(RUNS["bitcount_shots"] / "results.json")
    rows = r["rows"]
    shots_list = r["meta"]["shots"]
    fig, axes = plt.subplots(1, len(shots_list) + 1, figsize=(8.4, 2.7), sharey=True)
    exact = np.array([x["exact_kl_history"] for x in rows if x["shots"] is None])
    for ax, n in zip(axes[:-1], shots_list, strict=True):
        for crn, color, label in (
            (False, ORANGE, "independent draws"),
            (True, AQUA, "common random numbers"),
        ):
            hs = np.array(
                [x["exact_kl_history"] for x in rows if x["shots"] == n and x["crn"] == crn]
            )
            ax.plot(np.median(hs, axis=0), color=color, label=label)
            ax.fill_between(
                range(hs.shape[1]), hs.min(axis=0), hs.max(axis=0), color=color, alpha=0.15, lw=0
            )
        ax.plot(np.median(exact, axis=0), color=INK, ls="--", lw=1.1, label="exact probabilities")
        ax.set_yscale("log")
        ax.set_xlabel("sweep")
        ax.set_title(f"N = {n:,}", loc="left")
    axes[0].set_ylabel("exact KL of the iterate")
    fig.suptitle(
        "Bit-count KW–SA training when every loss evaluation is estimated from N shots (3 seeds: median, min–max)",
        fontsize=8.5,
        y=1.02,
    )
    axes[0].legend(loc="lower left", fontsize=6.5)
    ax = axes[-1]
    for crn, color, marker in ((False, ORANGE, "o"), (True, AQUA, "s")):
        med = [
            np.median([x["exact_kl_final"] for x in rows if x["shots"] == n and x["crn"] == crn])
            for n in shots_list
        ]
        ax.plot(shots_list, med, marker=marker, color=color)
    ax.plot(
        shots_list,
        [31 / (2 * 0.487 * n) for n in shots_list],
        color=PAPER,
        ls=":",
        label="plug-in KL bias",
    )
    ax.axhline(
        np.median([x["exact_kl_final"] for x in rows if x["shots"] is None]),
        color=INK,
        ls="--",
        lw=1.1,
    )
    ax.set_xscale("log")
    ax.set_xlabel("N")
    ax.set_title("after 20 sweeps", loc="left")
    ax.legend(loc="upper right", fontsize=6.5)
    save(fig, "fig_bitcount_sampled_training")


def fig_crn_variance() -> None:
    """Measured variance of the central difference with and without common random numbers."""
    net = bitcount.network(3, 2)
    rng = np.random.default_rng(0)
    x = net.random_params(rng)
    cs = [0.4, 0.2, 0.1, 0.05, 0.025]
    ratios, v_ind, v_crn = [], [], []
    for c in cs:
        e = np.zeros_like(x)
        e[0] = c
        be = ShotBackend(ref.StatevectorBackend(), shots=400, seed=1)
        loss = bitcount.kl_loss(net, be)
        gi, gc = [], []
        for _ in range(300):
            with be.common_random_numbers():
                gc.append((loss(x + e) - loss(x - e)) / (2 * c))
            gi.append((loss(x + e) - loss(x - e)) / (2 * c))
        v_ind.append(np.var(gi))
        v_crn.append(np.var(gc))
        ratios.append(np.var(gi) / np.var(gc))
    fig, ax = plt.subplots(figsize=(4.0, 2.9))
    ax.plot(cs, v_ind, marker="o", color=ORANGE, label="independent draws")
    ax.plot(cs, v_crn, marker="s", color=AQUA, label="common random numbers")
    for c, r_ in zip(cs, ratios, strict=True):
        ax.text(c, v_ind[cs.index(c)] * 1.4, f"×{r_:.0f}", ha="center", fontsize=6.5, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(cs)
    ax.set_xticklabels([f"{c:g}" for c in cs])
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.invert_xaxis()
    ax.set_xlabel("finite-difference step c (rad)")
    ax.set_ylabel("variance of the derivative estimate")
    ax.set_title("400 shots per probe, 300 repetitions; labels: variance ratio", loc="left")
    ax.legend(loc="upper left", fontsize=6.5)
    save(fig, "fig_crn_variance")


def fig_flappy_shots_programme() -> None:
    r = load(RUNS["flappy_shots"] / "results.json")
    rows = r["rows"]
    labels = []
    for x in rows:
        if x["label"] not in labels:
            labels.append(x["label"])
    colors = dict(zip(labels, [VIOLET, AQUA, BLUE, GREEN, ORANGE], strict=False))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    qs = ["0.01", "0.1", "0.25", "0.5", "0.75", "0.9"]
    ax = axes[0]
    for lab in labels:
        x = next(y for y in rows if y["label"] == lab and y["shots"] == 0)
        ax.plot(
            [float(q) for q in qs],
            [x["margin_quantiles"][q] for q in qs],
            marker="o",
            color=colors[lab],
            label=lab,
        )
    ax.set_yscale("log")
    ax.set_xlabel("quantile of played frames")
    ax.set_ylabel("decision margin δ")
    ax.legend(fontsize=6.5)
    ax.set_title("decision margins", loc="left")
    ax = axes[1]
    for lab in labels:
        fixed = sorted(
            (y for y in rows if y["label"] == lab and y["shots"] > 0 and y["sprt"] is None),
            key=lambda y: y["shots"],
        )
        ax.plot(
            [y["shots"] for y in fixed],
            [y["perfect"] for y in fixed],
            marker="o",
            color=colors[lab],
            label=f"{lab}, fixed N",
        )
        sprt = [y for y in rows if y["label"] == lab and y["sprt"] is not None]
        ax.scatter(
            [y["mean_shots_per_frame"] for y in sprt],
            [y["perfect"] for y in sprt],
            marker="*",
            s=90,
            color=colors[lab],
            edgecolor=INK,
            linewidths=0.6,
            zorder=5,
        )
    ax.set_xscale("log")
    pct(ax)
    ax.set_xlabel("shots per frame (stars: sequential test)")
    ax.set_ylabel("perfect games")
    ax.set_title("play quality vs budget", loc="left")
    ax = axes[2]
    for lab in labels:
        sprt = [y for y in rows if y["label"] == lab and y["sprt"] is not None]
        ax.scatter(
            [y["sprt"]["delta0"] for y in sprt],
            [y["mean_shots_per_frame"] for y in sprt],
            s=[30 if y["sprt"]["max_shots"] == 4000 else 60 for y in sprt],
            color=colors[lab],
            edgecolor=INK,
            linewidths=0.4,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("indifference margin δ₀ (small dot: cap 4 000, large: 10 000)")
    ax.set_ylabel("mean shots per frame")
    ax.set_title("sequential-test cost", loc="left")
    fig.tight_layout()
    save(fig, "fig_flappy_shots_programme")


def fig_sprt_cost() -> None:
    """Expected shots of the sequential test vs the true margin, against the fixed budget that
    reaches the same 5 % error — simulated with the repo's own rule."""
    rule = flappy_agent.SPRTDecision(delta0=0.02, alpha=0.05, beta=0.05, max_shots=10_000, block=20)
    rng = np.random.default_rng(0)
    deltas = np.array([0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.4])
    mean_shots, err = [], []
    for d in deltas:
        res = [rule.decide(0.5 + d, rng) for _ in range(300)]
        mean_shots.append(np.mean([n for _, n in res]))
        err.append(np.mean([not f for f, _ in res]))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    ax = axes[0]
    ax.plot(
        deltas[1:],
        mean_shots[1:],
        marker="o",
        color=AQUA,
        label="sequential test, δ₀ = 0.02, cap 10 000",
    )
    ax.plot(
        deltas[1:],
        (1.645 / (2 * deltas[1:])) ** 2,
        color=ORANGE,
        ls="--",
        label="fixed N for 5 % flip probability",
    )
    ax.axhline(mean_shots[0], color=INK2, ls=":", lw=1)
    ax.text(
        0.006, mean_shots[0] * 1.1, f"at δ = 0: {mean_shots[0]:.0f} shots", fontsize=6.5, color=INK2
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("true margin δ of the frame")
    ax.set_ylabel("mean shots to decide")
    ax.legend(fontsize=6.5, loc="upper right")
    ax = axes[1]
    ax.plot(deltas, err, marker="o", color=AQUA)
    ax.axhline(0.05, color=INK2, ls=":", lw=1)
    ax.axvline(0.02, color=INK2, ls=":", lw=1)
    ax.set_xlabel("true margin δ")
    ax.set_ylabel("wrong decisions")
    pct(ax)
    ax.set_title("error is bounded (5 %) only beyond δ₀", loc="left")
    save(fig, "fig_sprt_cost")


# --- main ----------------------------------------------------------------------------------------

ALL = [
    fig_activation_surface,
    fig_bitcount_histograms,
    fig_bitcount_convergence,
    fig_iris_losses,
    fig_iris_histograms,
    fig_flappy_policy,
    lambda: fig_ecg_trace(
        "ecg", "fig_ecg_trace", "Synthetic ECG, σ = 0.02 noise, 3 hidden neurons"
    ),
    lambda: fig_ecg_trace(
        "ecg_physionet", "fig_ecg_trace_physionet", "PhysioNet MIT-BIH record 100, 3 hidden neurons"
    ),
    fig_ecg_table,
    fig_bitcount_noise_devices,
    fig_bitcount_noise_scale,
    fig_bitcount_noise_aware,
    fig_bitcount_plugin_bias,
    fig_flappy_noise,
    fig_flappy_shot_noise,
    fig_ecg_iris_noise,
    fig_hardware_entanglers,
    fig_hardware_fidelity,
    fig_hardware_qpu_time,
    fig_bitcount_sampled_training,
    fig_crn_variance,
    fig_flappy_shots_programme,
    fig_sprt_cost,
]


def main() -> None:
    t0 = time.perf_counter()
    failures = []
    for f in ALL:
        try:
            f()
        except Exception as e:  # noqa: BLE001 — report every failure, then fail loudly
            log(f"FAILED {getattr(f, '__name__', 'lambda')}: {type(e).__name__}: {e}")
            failures.append((getattr(f, "__name__", "lambda"), repr(e)))
    log(
        f"{len(ALL) - len(failures)}/{len(ALL)} figures in {time.perf_counter() - t0:.0f} s → {OUT}"
    )
    if failures:
        raise SystemExit(f"{len(failures)} figure(s) failed: {failures}")


if __name__ == "__main__":
    main()
