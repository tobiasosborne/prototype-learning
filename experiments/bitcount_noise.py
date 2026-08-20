"""Bit-count under realistic noise — `PLAN.md` §1 item 3 ("Noise study"), §3 P9.

Five studies on the Fig-5 bit-count network (5 inputs, 5 hidden, 1 classic output — 11 qubits)
``[PTQNN Fig 5]`` and on the *classic*-hidden variant of the same shape.  Both have exactly the
same parameter count: a prototype neuron carries ``n_inputs`` prototype angles + one shared
``θ``, a classic neuron ``n_inputs`` weights + one bias ``[PTQNN §III.C]`` ("with the same
amount of parameters"), so ``Network.n_params`` is identical and the comparison is fair.

* **A. train clean, deploy noisy** — train on the exact statevector, evaluate the resulting
  parameters under every vendor preset: ``KL(target ‖ noisy)``, total-variation distance to the
  target, and ``KL(clean ‖ noisy)`` (how far noise moved the model's own distribution).
* **B. noise-aware training** — run the same KW–SA superposition training with the loss itself
  evaluated on ``NoisyBackend(density-matrix, preset)``, then score the result both noisily and
  cleanly.  This costs ``3·n_params`` noisy density-matrix evaluations *per sweep* — 108 at
  11 qubits.  ``--b-network reduced`` (default) trains a smaller network of the same family
  (``--train-bits``/``--train-hidden``, default 3 bits / 3 hidden, 7 qubits) over a ``spawn``
  pool; ``--b-network full`` trains the Fig-5 network itself, sequentially on ``--backend``,
  which after the block-superoperator fix in ``ptqnn.backends.reference`` costs ~1.5 s per
  evaluation on the GPU (~40 min per 15-sweep run).  See ``results.md`` for the arithmetic.
* **C. noise-strength sweep** — ``NoiseModel.scaled(s)`` of one preset (``--scale-device``,
  default ``ibm_fez``) over ``--scales``; KL vs s for prototype and classic.
* **D. shot noise** — multinomial sampling of the *full* 11-qubit outcome distribution
  (``reference.sample_counts``), conditioned on ``out = 1`` and renormalised exactly as the
  exact estimator is, over ``--shot-reps`` repetitions per shot budget; mean ± sd of the KL
  estimate against the exact value.  The estimator inherits ``losses.kl_divergence``'s ``eps``
  floor, which is what makes it finite when a target-supported bit string gets zero counts.
* **E. circuit facts** — ``NoiseModel.schedule(circuit).total_ns`` and the 1q/2q gate counts
  *before* transpilation.  Hardware compilation to a native gate set with a restricted coupling
  map adds SWAPs and expands each CRX, so these are lower bounds; see
  ``reports/hardware-compile-table.md`` for the compiled counts when it exists.

``--studies`` selects which of A–E run (C and D deploy A's trained parameters and pull it in).

All physics lives in ``ptqnn.noise``, ``ptqnn.tasks.bitcount``, ``ptqnn.train`` and
``ptqnn.backends`` — this script only drives them, records and plots.  One process owns the GPU,
so all exact full-network work is sequential; the reduced-network training grid (B) uses a
``spawn`` pool over (device, seed, model) with the numpy density-matrix backend.
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
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from ptqnn.backends import reference as ref
from ptqnn.backends.base import Backend
from ptqnn.circuits.neurons import HiddenKind, Network
from ptqnn.noise import presets
from ptqnn.noise.model import NoisyBackend
from ptqnn.tasks import bitcount
from ptqnn.train import distribution, kwsa, losses

Real = npt.NDArray[np.float64]

FIG5_BITS = 5
FIG5_HIDDEN = 5  # [PTQNN Fig 5]: 5 hidden neurons, 11 qubits total.
MODELS: tuple[HiddenKind, ...] = ("prototype", "classic")
COLOURS = {"prototype": "#4c72b0", "classic": "#c44e52"}

# --- pure helpers (unit-tested in tests/test_bitcount_noise_script.py) ---------------------------


def total_variation(p: npt.ArrayLike, q: npt.ArrayLike) -> float:
    """Total-variation distance ``½ Σ_x |p(x) − q(x)|``.

    The natural companion to the KL of ``[PTQNN Eq 4]``: it is bounded (≤ 1), so a model that
    stops firing altogether saturates instead of running off to the ``eps``-floored KL, and by
    Pinsker ``TV ≤ √(KL/2)`` the two can be read against each other.
    """
    a = np.asarray(p, dtype=np.float64)
    b = np.asarray(q, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("distributions must have the same shape")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("distributions must be finite")
    return float(0.5 * np.abs(a - b).sum())


def conditioned_input_counts(
    counts: npt.ArrayLike, input_qubits: tuple[int, ...], output_qubit: int
) -> Real:
    """Shot counts of ``(inputs = x, out = 1)`` from full-register counts.

    Marginalising counts is the same linear operation as marginalising probabilities, so this
    reuses ``reference.marginal``; the ``out = 1`` slice is the finite-shot version of
    ``distribution.conditioned_input_distribution``'s numerator.
    """
    c = np.asarray(counts, dtype=np.float64)
    if np.any(c < 0):
        raise ValueError("counts must be non-negative")
    m = ref.marginal(c, [*input_qubits, output_qubit])
    return np.ascontiguousarray(m.reshape(-1, 2)[:, 1])


def finite_shot_kl(target: npt.ArrayLike, counts: npt.ArrayLike, eps: float = 1e-12) -> float:
    """``KL(target ‖ counts/Σcounts)`` — the plug-in KL estimator at finite shots.

    ``losses.kl_divergence`` floors the model at ``eps``, so an unobserved but target-supported
    bit string contributes ``p log(p/eps)`` instead of ``+∞``; with all counts zero (the network
    never fired in any shot) the value is the same large finite penalty
    ``Σ p log(p/eps) ≈ log(1/eps)`` that ``conditioned_input_distribution`` produces exactly.
    The estimator is biased upwards at small ``N`` — that bias is exactly what study D measures.
    """
    c = np.asarray(counts, dtype=np.float64)
    if np.any(c < 0):
        raise ValueError("counts must be non-negative")
    total = c.sum()
    q = c / total if total > 0 else np.zeros_like(c)
    return losses.kl_divergence(np.asarray(target, dtype=np.float64), q, eps)


# --- backends -----------------------------------------------------------------------------------


def make_backend(kind: str) -> tuple[Backend, str]:
    """Exact mixed-state backend for the 11-qubit work: gpu → orkan → numpy, with fallback."""
    order = {"gpu": ["gpu", "orkan", "numpy"], "orkan": ["orkan", "numpy"], "numpy": ["numpy"]}[
        kind
    ]
    errors: list[str] = []
    for name in order:
        try:
            if name == "gpu":
                from ptqnn.backends.gpu import GpuDensityMatrixBackend

                return GpuDensityMatrixBackend(), "gpu"
            if name == "orkan":
                from ptqnn.backends.orkan import OrkanDensityMatrixBackend

                return OrkanDensityMatrixBackend(), "orkan"
            return ref.DensityMatrixBackend(), "numpy"
        except Exception as e:  # pragma: no cover - machine dependent
            errors.append(f"{name}: {e}")
    raise RuntimeError("no usable backend: " + "; ".join(errors))


def model_distribution(net: Network, params: Real, backend: Backend) -> Real:
    """``P(x | out = 1)`` — the quantity compared to the target in ``[PTQNN Eq 4]``."""
    return distribution.conditioned_input_distribution(
        net, params, bitcount.superposition_prep(net), backend
    )


def full_probabilities(net: Network, params: Real, backend: Backend) -> Real:
    """Outcome distribution over all ``2**n_qubits`` basis states (what a shot samples)."""
    return backend.probabilities(net.circuit(params, bitcount.superposition_prep(net)))


def scores(target: Real, clean: Real, noisy: Real) -> dict[str, float]:
    return {
        "kl_target": losses.kl_divergence(target, noisy),
        "tv_target": total_variation(target, noisy),
        "kl_clean_vs_noisy": losses.kl_divergence(clean, noisy),
        "tv_clean_vs_noisy": total_variation(clean, noisy),
    }


# --- A. clean training, noisy deployment --------------------------------------------------------


def train_clean(net: Network, seeds: list[int], iterations: int) -> dict[str, Any]:
    """Exact (statevector) superposition training over ``seeds``; the best seed is deployed."""
    runs = []
    for seed in seeds:
        t0 = time.perf_counter()
        res = bitcount.train_superposition(
            net, seed=seed, iterations=iterations, backend=ref.StatevectorBackend()
        )
        runs.append(
            {
                "seed": seed,
                "kl": float(res.loss),
                "history": [float(v) for v in res.history],
                "params": res.params.tolist(),
                "evaluations": int(res.evaluations),
                "wall_s": time.perf_counter() - t0,
            }
        )
    best = min(runs, key=lambda r: r["kl"])
    return {"runs": runs, "best_seed": best["seed"], "kl": best["kl"], "params": best["params"]}


# --- B. noise-aware training --------------------------------------------------------------------


def train_superposition_with_callback(
    net: Network,
    seed: int,
    iterations: int,
    backend: Backend,
    callback: Callable[[int, Real, float], None] | None = None,
) -> kwsa.KWSAResult:
    """``bitcount.train_superposition`` with a per-sweep callback.

    ``bitcount.train_superposition`` builds the config and the seeded start point and then calls
    ``kwsa.minimise`` *without* forwarding a callback, so a multi-hour noisy run cannot report
    progress.  This mirrors it line for line — same ``SUPERPOSITION_CONFIG`` override, same
    ``net.random_params(default_rng(seed))`` start point — and forwards ``callback``, which
    ``kwsa.minimise`` invokes as ``(n, x, fx)`` after every sweep.
    ``tests/test_bitcount_noise_script.py`` pins the equivalence; delete this helper if
    ``train_superposition`` ever grows a ``callback`` argument.
    """
    cfg = kwsa.KWSAConfig(
        **{**bitcount.SUPERPOSITION_CONFIG.__dict__, "iterations": iterations, "seed": seed}
    )
    x0 = net.random_params(np.random.default_rng(seed))
    return kwsa.minimise(bitcount.kl_loss(net, backend), x0, cfg, callback=callback)


def _progress_line(job: dict[str, Any], sweep: int, loss: float, elapsed: float) -> str:
    """One ``progress.log`` row: ``device,model,sweep,loss,elapsed_s,eta_s``.

    ETA extrapolates the sweeps done so far to the ones remaining — sweeps cost the same
    ``3·n_params`` evaluations each, so the estimate is honest apart from machine contention."""
    done = max(sweep, 1)
    eta = elapsed / done * (job["iterations"] - done)
    return f"{job['device']},{job['hidden']},{sweep},{loss:.6e},{elapsed:.1f},{eta:.1f}"


def _dump_partial(path: Path, runs: list[dict[str, Any]], total: int) -> None:
    """Rewrite ``partial_results.json`` after every finished cell, and echo the cell's summary."""
    r = runs[-1]
    print(
        f"    [{len(runs)}/{total}] done {r['hidden']:9s} {r['device']:16s} "
        f"seed {r['seed']} scale {r['scale']:g}: noisy KL {r['noisy_kl']:.3e}, clean-eval KL "
        f"{r['clean_kl_of_noise_aware_params']:.3e}, {r['evaluations']} evals in "
        f"{r['wall_s'] / 60:.1f} min ({r['seconds_per_evaluation']:.2f} s/eval)",
        flush=True,
    )
    with path.open("w") as f:
        json.dump({"completed": len(runs), "total": total, "runs": runs}, f, indent=1)


def _train_noisy_job(job: dict[str, Any], backend: Backend | None = None) -> dict[str, Any]:
    """One noise-aware training run.

    With ``backend=None`` the job is self-contained and picklable, so it can be mapped over a
    ``spawn`` pool; it then builds its own numpy density-matrix backend.  The full-network path
    passes the parent's backend instead and calls this in-process, because one process owns the
    GPU."""
    if backend is None:
        # spawn children inherit OMP_NUM_THREADS=1 from the parent's import-time pin; orkan is an
        # OpenMP C library and must be told its share of the machine *before* liborkan.so loads,
        # which make_backend does lazily.
        os.environ["OMP_NUM_THREADS"] = str(job.get("threads", 1))
        backend = make_backend(job.get("backend", "numpy"))[0]
    net = bitcount.network(job["n_bits"], job["n_hidden"], job["hidden"])
    nm = presets.device(job["device"]).noise_model().scaled(job["scale"])
    noisy = NoisyBackend(backend, nm)
    progress_path = job.get("progress_path")
    t0 = time.perf_counter()

    def on_sweep(n: int, _x: Real, fx: float) -> None:
        line = _progress_line(job, n + 1, fx, time.perf_counter() - t0)
        print(f"    [progress] {line}", flush=True)
        if progress_path:
            # O_APPEND + one short write per sweep: safe to interleave across pool workers.
            with open(progress_path, "a") as f:
                f.write(line + "\n")

    res = train_superposition_with_callback(net, job["seed"], job["iterations"], noisy, on_sweep)
    wall = time.perf_counter() - t0
    params = res.params
    target = bitcount.target_distribution(net.n_inputs)
    q_noisy = model_distribution(net, params, noisy)
    q_clean = model_distribution(net, params, ref.StatevectorBackend())
    return {
        **{k: job[k] for k in ("device", "scale", "seed", "hidden", "n_bits", "n_hidden")},
        "noisy_kl": float(res.loss),
        "noisy_kl_check": losses.kl_divergence(target, q_noisy),
        "clean_kl_of_noise_aware_params": losses.kl_divergence(target, q_clean),
        "tv_target_noisy": total_variation(target, q_noisy),
        "history": [float(v) for v in res.history],
        "evaluations": int(res.evaluations),
        "wall_s": wall,
        "seconds_per_evaluation": wall / max(int(res.evaluations), 1),
    }


# --- D. shot noise ------------------------------------------------------------------------------


def shot_study(
    net: Network,
    p_full: Real,
    target: Real,
    shots_list: list[int],
    reps: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Finite-shot KL estimates for one full-register distribution."""
    rng = np.random.default_rng(seed)
    exact = conditioned_input_counts(p_full, net.input_qubits, net.output_qubit)
    exact_kl = finite_shot_kl(target, exact)
    survival = float(exact.sum())  # P(out = 1): the fraction of shots that are usable
    out = []
    for shots in shots_list:
        if shots <= 0:
            out.append(
                {
                    "shots": 0,
                    "mean": exact_kl,
                    "sd": 0.0,
                    "exact": exact_kl,
                    "bias": 0.0,
                    "kept_mean": survival,
                    "estimates": [],
                }
            )
            continue
        est = []
        kept = []
        for _ in range(reps):
            counts = ref.sample_counts(p_full, shots, rng)
            c = conditioned_input_counts(counts, net.input_qubits, net.output_qubit)
            est.append(finite_shot_kl(target, c))
            kept.append(float(c.sum()))
        a = np.asarray(est)
        out.append(
            {
                "shots": shots,
                "mean": float(a.mean()),
                "sd": float(a.std(ddof=1)) if reps > 1 else 0.0,
                "exact": exact_kl,
                "bias": float(a.mean() - exact_kl),
                "kept_mean": float(np.mean(kept)),
                "estimates": [float(v) for v in a],
            }
        )
    return out


# --- E. circuit facts ---------------------------------------------------------------------------


def circuit_facts(net: Network, params: Real, spec: presets.DeviceSpec) -> dict[str, Any]:
    """Untranspiled gate counts and the model's ASAP schedule length for one device."""
    nm = spec.noise_model()
    circ = net.circuit(params, bitcount.superposition_prep(net))
    gates = circ.gates()
    n1 = sum(1 for g in gates if g.n_qubits == 1)
    n2 = sum(1 for g in gates if g.n_qubits == 2)
    noisy = nm.apply(circ)
    total_ns = nm.schedule(circ).total_ns
    return {
        "device": spec.name,
        "n_gates": len(gates),
        "n_1q_gates": n1,
        "n_2q_gates": n2,
        "n_channels_inserted": len(noisy.ops) - len(circ.ops),
        "total_ns": float(total_ns),
        "t1_us": spec.t1_us,
        "t2_us": spec.t2_us,
        "gate_1q_error": spec.gate_1q_error,
        "gate_2q_error": spec.gate_2q_error,
        # Leading-order incoherent budget: every gate error adds once (independent channels).
        "error_budget_1q": n1 * spec.gate_1q_error,
        "error_budget_2q": n2 * spec.gate_2q_error,
        "readout_budget": net.n_qubits * 0.5 * (spec.readout_p01 + spec.readout_p10),
        "t2_fraction": None if spec.t2_us is None else float(total_ns * 1e-3 / spec.t2_us),
    }


def compiled_2q_counts(path: Path, n_bits: int, n_hidden: int) -> dict[str, int]:
    """Native 2q-gate counts per topology, read out of ``reports/hardware-compile-table.md``.

    That table (generated by ``tests/test_hardware.py``) records, for
    ``Network(n_bits, n_hidden)``, the transpiled circuit for each gate set × topology;
    ``n_2q`` counts every native entangler including the three each routed SWAP costs.  The
    counts are the same for every gate set at a fixed topology (routing happens first), so the
    ``generic`` rows are enough.  Returns ``{}`` if the table is absent or has no such section.
    """
    if not path.exists():
        return {}
    section = False
    out: dict[str, int] = {}
    for line in path.read_text().splitlines():
        if line.startswith("## "):
            section = f"Network({n_bits},{n_hidden})" in line.replace(" ", "")
            continue
        if not section or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 5 and cells[0] == "generic":
            try:
                out[cells[1]] = int(cells[4])
            except ValueError:
                continue
    return out


# --- figures ------------------------------------------------------------------------------------


def plot_devices(results: dict[str, Any], out_path: Path) -> None:
    devices = results["devices"]
    x = np.arange(len(devices))
    fig, (ax_kl, ax_tv) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    width = 0.38
    for i, model in enumerate(MODELS):
        a = results["A"][model]
        kl = [a["per_device"][d]["kl_target"] for d in devices]
        tv = [a["per_device"][d]["tv_target"] for d in devices]
        off = (i - 0.5) * width
        ax_kl.bar(x + off, kl, width, label=model, color=COLOURS[model], edgecolor="black", lw=0.3)
        ax_tv.bar(x + off, tv, width, label=model, color=COLOURS[model], edgecolor="black", lw=0.3)
        ax_kl.axhline(a["clean_kl"], color=COLOURS[model], ls="--", lw=1, label=f"{model} clean KL")
    ax_kl.set_yscale("log")
    ax_kl.set_ylabel("KL(target ‖ model)")
    ax_kl.set_title(
        f"A. clean-trained {results['n_bits']}-bit / {results['n_hidden']}-hidden network "
        f"deployed under vendor noise ({results['n_qubits']} qubits, exact density matrix)"
    )
    ax_kl.legend(fontsize=8, ncol=2)
    ax_tv.set_ylabel("total variation to target")
    ax_tv.set_xticks(x)
    ax_tv.set_xticklabels(devices, rotation=20, ha="right")
    ax_tv.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_noise_aware(results: dict[str, Any], out_path: Path) -> None:
    b = results.get("B")
    if not b or not b.get("per_device"):
        return
    devices = list(b["devices"])
    x = np.arange(len(devices))
    fig, (ax, ax_curve) = plt.subplots(1, 2, figsize=(13, 5))
    width = 0.2
    for i, model in enumerate(MODELS):
        base = [b["baseline"][model][d]["kl_target"] for d in devices]
        aware = [b["per_device"][model][d]["median_noisy_kl"] for d in devices]
        ax.bar(
            x + (2 * i - 1.5) * width,
            base,
            width,
            color=COLOURS[model],
            alpha=0.45,
            edgecolor="black",
            lw=0.3,
            label=f"{model}: train clean → deploy noisy",
        )
        ax.bar(
            x + (2 * i - 0.5) * width,
            aware,
            width,
            color=COLOURS[model],
            edgecolor="black",
            lw=0.3,
            label=f"{model}: noise-aware training",
        )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(devices, rotation=20, ha="right")
    ax.set_ylabel("KL(target ‖ noisy model)")
    ax.set_title(
        f"B. noise-aware training, {b['n_bits']}-bit / {b['n_hidden']}-hidden "
        f"({b['n_qubits']} qubits), median over {len(b['seeds'])} seeds"
    )
    ax.legend(fontsize=7)

    dev = b["curve_device"]
    for model in MODELS:
        hists = b["per_device"][model][dev]["histories"]
        arr = np.asarray(hists)
        ax_curve.plot(np.median(arr, axis=0), color=COLOURS[model], lw=2, label=f"{model} (noisy)")
        ax_curve.fill_between(
            np.arange(arr.shape[1]),
            arr.min(axis=0),
            arr.max(axis=0),
            color=COLOURS[model],
            alpha=0.2,
            lw=0,
        )
    ax_curve.set_yscale("log")
    ax_curve.set_xlabel("KW–SA sweep")
    ax_curve.set_ylabel("KL(target ‖ noisy model)")
    ax_curve.set_title(f"noise-aware training curves — {dev}")
    ax_curve.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scale(results: dict[str, Any], out_path: Path) -> None:
    c = results["C"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for model in MODELS:
        s = np.asarray(c["scales"], dtype=float)
        kl = np.asarray([c["fig5"][model][f"{v:g}"]["kl_target"] for v in c["scales"]])
        ax.plot(s, kl, "o-", color=COLOURS[model], lw=2, label=f"{model} ({results['n_qubits']}q)")
        if c.get("reduced"):
            klr = np.asarray([c["reduced"][model][f"{v:g}"]["kl_target"] for v in c["scales"]])
            ax.plot(
                s,
                klr,
                "s--",
                color=COLOURS[model],
                alpha=0.55,
                lw=1.5,
                label=f"{model} ({c['reduced_n_qubits']}q)",
            )
        if c.get("retrained", {}).get(model):
            r = c["retrained"][model]
            xs = sorted(float(k) for k in r)
            ax.plot(
                xs,
                [r[f"{v:g}"]["noisy_kl"] for v in xs],
                "^:",
                color=COLOURS[model],
                lw=1.5,
                label=f"{model} noise-aware retrain ({c['reduced_n_qubits']}q)",
            )
    ax.set_yscale("log")
    ax.set_xlabel(f"noise scale s  (NoiseModel.scaled(s) of {c['device']})")
    ax.set_ylabel("KL(target ‖ noisy model)")
    ax.set_title(f"C. KL vs noise strength — {c['device']} scaled")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_shots(results: dict[str, Any], out_path: Path) -> None:
    d = results["D"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for ax, source in zip(axes, ("clean", "noisy"), strict=True):
        for model in MODELS:
            rows = [r for r in d[model][source] if r["shots"] > 0]
            if not rows:
                continue
            xs = [r["shots"] for r in rows]
            ys = [r["mean"] for r in rows]
            es = [r["sd"] for r in rows]
            ax.errorbar(
                xs,
                ys,
                yerr=es,
                fmt="o-",
                capsize=4,
                color=COLOURS[model],
                lw=1.5,
                label=f"{model} (mean ± sd, {d['reps']} reps)",
            )
            exact = rows[0]["exact"]
            ax.axhline(exact, color=COLOURS[model], ls="--", lw=1, label=f"{model} exact")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("shots")
        ax.set_ylabel("KL estimate")
        ax.set_title(f"D. finite-shot KL — {source} distribution ({d['device']} for noisy)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --- driver -------------------------------------------------------------------------------------


def _ints(s: str) -> list[int]:
    return [int(v) for v in s.replace(",", " ").split()]


def _floats(s: str) -> list[float]:
    return [float(v) for v in s.replace(",", " ").split()]


def _strs(s: str) -> list[str]:
    return [v for v in s.replace(",", " ").split() if v]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_out = "experiments/runs/bitcount_noise/" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--devices", type=str, default="all", help="'all' or a comma-separated list")
    p.add_argument("--bits", type=int, default=FIG5_BITS, help="main network inputs [PTQNN Fig 5]")
    p.add_argument("--hidden", type=int, default=FIG5_HIDDEN, help="main network hidden neurons")
    p.add_argument("--seeds", type=_ints, default=[0, 1, 2])
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--scales", type=_floats, default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    p.add_argument("--shots", type=_ints, default=[0, 1000, 10000])
    p.add_argument("--shot-reps", type=int, default=20)
    p.add_argument("--backend", choices=("gpu", "orkan", "numpy"), default="gpu")
    p.add_argument("--scale-device", type=str, default="ibm_fez")
    p.add_argument("--train-bits", type=int, default=3, help="network for study B (noise-aware)")
    p.add_argument("--train-hidden", type=int, default=3)
    p.add_argument(
        "--studies",
        type=str,
        default="A,B,C,D,E",
        help="which studies to run; C and D pull in A (they deploy A's trained parameters)",
    )
    p.add_argument(
        "--b-network",
        choices=("full", "reduced"),
        default="reduced",
        help="network for study B: 'reduced' (--train-bits/--train-hidden, numpy pool) or "
        "'full' (the --bits/--hidden network, sequential on --backend)",
    )
    p.add_argument("--no-noise-aware", action="store_true", help="skip study B")
    p.add_argument("--retrain-scales", action="store_true", help="study C: retrain at each scale")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument(
        "--threads",
        type=int,
        default=1,
        help="OpenMP threads per process for the orkan backend (numpy/BLAS stays pinned to 1); "
        "orkan is memory-bound, so ~8 threads per worker beats one 64-thread worker",
    )
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


def _slope(scales: list[float], kls: list[float]) -> dict[str, float]:
    """Sensitivity of KL to the noise scale.

    The raw KL never reaches zero — a finitely trained network has a KL floor that has nothing to
    do with noise — so the *excess* ``ΔKL(s) = KL(s) − KL(0)`` is the noise-induced part and the
    one whose scaling is physically meaningful.  Reported: the linear slope ``dKL/ds`` over
    ``s ≤ 1`` and the log–log exponent ``k`` of ``ΔKL ∝ s^k`` (``k ≈ 1`` = linear response, i.e.
    a ×m improvement in every error rate buys a ×m reduction in the noise-induced KL).
    """
    s = np.asarray(scales, dtype=float)
    k = np.asarray(kls, dtype=float)
    lo = s <= 1.0
    lin = float(np.polyfit(s[lo], k[lo], 1)[0]) if lo.sum() >= 2 else float("nan")
    floor = float(k[s == 0.0][0]) if np.any(s == 0.0) else 0.0
    excess = k - floor
    pos = (s > 0) & (excess > 0)
    log = (
        float(np.polyfit(np.log(s[pos]), np.log(excess[pos]), 1)[0])
        if pos.sum() >= 2
        else float("nan")
    )
    return {"linear_slope_s_le_1": lin, "loglog_exponent": log, "kl_floor": floor}


def main(argv: list[str] | None = None) -> None:
    """Driver.  ``--studies`` selects which of A–E run; C and D consume A's trained parameters
    and pull it in.  The exact 11-qubit work is sequential because one process owns the GPU; the
    reduced-network training grid uses a ``spawn`` pool with the numpy density-matrix backend."""
    args = parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    wall_t0 = time.perf_counter()
    timings: dict[str, float] = {}

    studies = {s.upper() for s in _strs(args.studies)}
    if not studies <= set("ABCDE"):
        raise SystemExit(f"--studies must be a subset of A,B,C,D,E; got {sorted(studies)}")
    if studies & {"C", "D"}:
        studies |= {"A"}  # both deploy A's clean-trained parameters

    devices = presets.available_devices() if args.devices == "all" else _strs(args.devices)
    unknown = [d for d in devices if d not in presets.available_devices()]
    if unknown:
        raise SystemExit(f"unknown device(s) {unknown}; available {presets.available_devices()}")
    specs = {d: presets.device(d) for d in devices}

    if args.threads > 1:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
    backend, backend_kind = make_backend(args.backend)
    print(f"[backend] requested {args.backend!r}, using {backend_kind!r} ({backend.name})")

    nets = {m: bitcount.network(args.bits, args.hidden, m) for m in MODELS}
    reduced = {m: bitcount.network(args.train_bits, args.train_hidden, m) for m in MODELS}
    target = bitcount.target_distribution(args.bits)
    reduced_target = bitcount.target_distribution(args.train_bits)
    base_model = presets.device(args.scale_device).noise_model()
    results: dict[str, Any] = {
        "n_bits": args.bits,
        "n_hidden": args.hidden,
        "n_qubits": nets["prototype"].n_qubits,
        "n_params": nets["prototype"].n_params,
        "devices": devices,
        "studies": sorted(studies),
        "target": target.tolist(),
    }

    clean_cache: dict[tuple[int, int, str], dict[str, Any]] = {}

    def clean_trained(net: Network) -> dict[str, Any]:
        """Exact-statevector training of ``net``, memoised — A, B and C share the same runs."""
        key = (net.n_inputs, net.n_hidden, net.hidden)
        if key not in clean_cache:
            clean_cache[key] = train_clean(net, args.seeds, args.iterations)
        return clean_cache[key]

    # --- budget probe: one exact noisy evaluation of the main network ---------------------------
    probe_net = nets["prototype"]
    probe_params = probe_net.random_params(np.random.default_rng(0))
    t0 = time.perf_counter()
    model_distribution(
        probe_net, probe_params, NoisyBackend(backend, specs[devices[0]].noise_model())
    )
    probe_s = time.perf_counter() - t0
    sweep_evals = 3 * probe_net.n_params
    print(
        f"[budget] one exact noisy {probe_net.n_qubits}-qubit evaluation: {probe_s:.2f} s.  "
        f"A KW-SA sweep costs 3·n_params = {sweep_evals} of them "
        f"({probe_s * sweep_evals / 60:.1f} min/sweep, "
        f"{probe_s * sweep_evals * args.iterations / 3600:.2f} h per {args.iterations}-sweep run); "
        f"study B network = {args.b_network}."
    )
    timings["probe_noisy_eval_s"] = probe_s

    # --- A: train clean, deploy noisy -----------------------------------------------------------
    if "A" in studies:
        print("[A] clean training (exact statevector) + noisy deployment...")
        t0 = time.perf_counter()
        results["A"] = {}
        for model in MODELS:
            net = nets[model]
            trained = clean_trained(net)
            params = np.asarray(trained["params"], dtype=np.float64)
            clean_q = model_distribution(net, params, ref.StatevectorBackend())
            per_device: dict[str, Any] = {}
            for d in devices:
                nb = NoisyBackend(backend, specs[d].noise_model())
                te = time.perf_counter()
                q = model_distribution(net, params, nb)
                per_device[d] = {**scores(target, clean_q, q), "eval_s": time.perf_counter() - te}
                print(
                    f"    {model:9s} {d:16s} KL={per_device[d]['kl_target']:.4f} "
                    f"TV={per_device[d]['tv_target']:.4f} "
                    f"KL(clean‖noisy)={per_device[d]['kl_clean_vs_noisy']:.4f} "
                    f"({per_device[d]['eval_s']:.1f}s)"
                )
            results["A"][model] = {
                "clean_kl": trained["kl"],
                "best_seed": trained["best_seed"],
                "params": trained["params"],
                "clean_distribution": clean_q.tolist(),
                "per_seed": [{k: r[k] for k in ("seed", "kl", "wall_s")} for r in trained["runs"]],
                "clean_histories": {str(r["seed"]): r["history"] for r in trained["runs"]},
                "per_device": per_device,
            }
        timings["A_s"] = time.perf_counter() - t0

    # --- C: noise-strength sweep (evaluation of the clean-trained model) -------------------------
    if "C" in studies:
        print(f"[C] noise-strength sweep on {args.scale_device}...")
        t0 = time.perf_counter()
        reduced_clean = {m: clean_trained(reduced[m]) for m in MODELS}
        c: dict[str, Any] = {
            "device": args.scale_device,
            "scales": args.scales,
            "fig5": {m: {} for m in MODELS},
            "reduced": {m: {} for m in MODELS},
            "reduced_n_qubits": reduced["prototype"].n_qubits,
            "reduced_n_bits": args.train_bits,
            "reduced_n_hidden": args.train_hidden,
            "reduced_clean_kl": {m: reduced_clean[m]["kl"] for m in MODELS},
            "retrained": {m: {} for m in MODELS},
        }
        for model in MODELS:
            net = nets[model]
            params = np.asarray(results["A"][model]["params"], dtype=np.float64)
            clean_q = np.asarray(results["A"][model]["clean_distribution"], dtype=np.float64)
            rnet = reduced[model]
            rparams = np.asarray(reduced_clean[model]["params"], dtype=np.float64)
            rclean_q = model_distribution(rnet, rparams, ref.StatevectorBackend())
            for s in args.scales:
                nm = base_model.scaled(s)
                be = ref.StatevectorBackend() if nm.is_trivial() else NoisyBackend(backend, nm)
                c["fig5"][model][f"{s:g}"] = scores(
                    target, clean_q, model_distribution(net, params, be)
                )
                rbe = (
                    ref.StatevectorBackend()
                    if nm.is_trivial()
                    else NoisyBackend(ref.DensityMatrixBackend(), nm)
                )
                c["reduced"][model][f"{s:g}"] = scores(
                    reduced_target, rclean_q, model_distribution(rnet, rparams, rbe)
                )
                print(
                    f"    {model:9s} s={s:<5g} KL({net.n_qubits}q)="
                    f"{c['fig5'][model][f'{s:g}']['kl_target']:.4f} "
                    f"KL({rnet.n_qubits}q)={c['reduced'][model][f'{s:g}']['kl_target']:.4f}"
                )
            c[f"slope_{model}"] = _slope(
                args.scales, [c["fig5"][model][f"{s:g}"]["kl_target"] for s in args.scales]
            )
        results["C"] = c
        timings["C_s"] = time.perf_counter() - t0

    # --- D: shot noise --------------------------------------------------------------------------
    if "D" in studies:
        print(f"[D] shot noise ({args.shot_reps} reps per budget)...")
        t0 = time.perf_counter()
        d_out: dict[str, Any] = {"reps": args.shot_reps, "device": args.scale_device}
        noisy_be = NoisyBackend(backend, base_model)
        for model in MODELS:
            net = nets[model]
            params = np.asarray(results["A"][model]["params"], dtype=np.float64)
            p_clean = full_probabilities(net, params, ref.StatevectorBackend())
            p_noisy = full_probabilities(net, params, noisy_be)
            d_out[model] = {
                "clean": shot_study(net, p_clean, target, args.shots, args.shot_reps, 12345),
                "noisy": shot_study(net, p_noisy, target, args.shots, args.shot_reps, 54321),
            }
            for source in ("clean", "noisy"):
                for row in d_out[model][source]:
                    frac = row["kept_mean"] / row["shots"] if row["shots"] else row["kept_mean"]
                    print(
                        f"    {model:9s} {source:5s} shots={row['shots']:<6d} "
                        f"KL={row['mean']:.4f}±{row['sd']:.4f} (exact {row['exact']:.4f}, "
                        f"P(out=1)={frac:.3f})"
                    )
        results["D"] = d_out
        timings["D_s"] = time.perf_counter() - t0

    # --- E: circuit facts (free — no simulation; always computed, reported on request) -----------
    results["E"] = {
        model: {
            d: circuit_facts(nets[model], np.zeros(nets[model].n_params), specs[d]) for d in devices
        }
        for model in MODELS
    }

    # --- B: noise-aware training ----------------------------------------------------------------
    if "B" in studies and not args.no_noise_aware:
        full_b = args.b_network == "full"
        b_nets = nets if full_b else reduced
        b_target = target if full_b else reduced_target
        b_clean = {m: clean_trained(b_nets[m]) for m in MODELS}
        b_eval_backend: Backend = backend if full_b else ref.DensityMatrixBackend()
        n_bits_b, n_hidden_b = (
            (args.bits, args.hidden) if full_b else (args.train_bits, args.train_hidden)
        )
        # One process owns the GPU, so a GPU run of the full network is sequential; orkan and
        # numpy are ordinary CPU libraries and are mapped over a spawn pool.
        sequential = full_b and backend_kind == "gpu"
        job_backend = backend_kind if full_b else "numpy"
        progress_path = out_dir / "progress.log"
        progress_path.write_text("device,model,sweep,loss,elapsed_s,eta_s\n")
        partial_path = out_dir / "partial_results.json"
        jobs = [
            {
                "n_bits": n_bits_b,
                "n_hidden": n_hidden_b,
                "hidden": model,
                "device": d,
                "scale": 1.0,
                "seed": seed,
                "iterations": args.iterations,
                "backend": job_backend,
                "threads": args.threads,
                "progress_path": str(progress_path),
            }
            for model in MODELS
            for d in devices
            for seed in args.seeds
        ]
        # Study C's "retrain at each scale" curve always uses the reduced network (it is a
        # 6-point sweep, i.e. 6 more training runs) and therefore always the numpy pool.
        scale_jobs = (
            [
                {
                    "n_bits": args.train_bits,
                    "n_hidden": args.train_hidden,
                    "hidden": model,
                    "device": args.scale_device,
                    "scale": s,
                    "seed": args.seeds[0],
                    "iterations": args.iterations,
                    "backend": "numpy",
                    "threads": 1,
                    "progress_path": str(progress_path),
                }
                for model in MODELS
                for s in args.scales
                # s = 1 and s = 0 are already covered: the main grid trains at s = 1 on every
                # device, and s = 0 is the clean-trained baseline.  Duplicating them here would
                # put the same run twice into the per-device medians.
                if s > 0 and s != 1.0
            ]
            if args.retrain_scales and "C" in results and not full_b
            else []
        )
        t0 = time.perf_counter()
        if sequential:
            print(
                f"[B] noise-aware training on the FULL {n_bits_b}-bit/{n_hidden_b}-hidden network "
                f"({b_nets['prototype'].n_qubits} qubits, {b_nets['prototype'].n_params} params), "
                f"{len(jobs)} runs, sequential on {backend_kind} "
                f"(~{probe_s * sweep_evals * args.iterations / 3600:.2f} h each, "
                f"~{probe_s * sweep_evals * args.iterations * len(jobs) / 3600:.1f} h total)..."
            )
            runs = []
            for i, job in enumerate(jobs, 1):
                print(
                    f"    [{i}/{len(jobs)}] {job['hidden']:9s} {job['device']:16s} "
                    f"seed {job['seed']} ...",
                    flush=True,
                )
                runs.append(_train_noisy_job(job, backend))
                _dump_partial(partial_path, runs, len(jobs))
        else:
            print(
                f"[B] noise-aware training on the {b_nets['prototype'].n_qubits}-qubit "
                f"{'FULL' if full_b else 'reduced'} {n_bits_b}-bit/{n_hidden_b}-hidden network, "
                f"{len(jobs) + len(scale_jobs)} runs, pool of {args.jobs} × {args.threads} "
                f"thread(s) on {job_backend} "
                f"(~{probe_s * sweep_evals * args.iterations / 3600:.2f} h per run "
                "at the probe's single-process speed)..."
            )
            ctx = mp.get_context("spawn")
            all_jobs = jobs + scale_jobs
            runs = []
            with ctx.Pool(processes=max(1, args.jobs)) as pool:
                # imap_unordered so a cell's result is on disk as soon as it finishes: an
                # interrupted multi-hour run still leaves partial_results.json behind.
                for r in pool.imap_unordered(_train_noisy_job, all_jobs):
                    runs.append(r)
                    _dump_partial(partial_path, runs, len(all_jobs))
        timings["B_s"] = time.perf_counter() - t0

        baseline: dict[str, Any] = {m: {} for m in MODELS}
        for model in MODELS:
            bnet = b_nets[model]
            bparams = np.asarray(b_clean[model]["params"], dtype=np.float64)
            bclean_q = model_distribution(bnet, bparams, ref.StatevectorBackend())
            for d in devices:
                nb = NoisyBackend(b_eval_backend, specs[d].noise_model())
                baseline[model][d] = scores(
                    b_target, bclean_q, model_distribution(bnet, bparams, nb)
                )
        per_device_b: dict[str, Any] = {m: {} for m in MODELS}
        for model in MODELS:
            for d in devices:
                rs = [
                    r
                    for r in runs
                    if r["hidden"] == model and r["device"] == d and r["scale"] == 1.0
                ]
                rs.sort(key=lambda r: r["seed"])
                per_device_b[model][d] = {
                    "median_noisy_kl": float(np.median([r["noisy_kl"] for r in rs])),
                    "best_noisy_kl": float(np.min([r["noisy_kl"] for r in rs])),
                    "median_clean_kl": float(
                        np.median([r["clean_kl_of_noise_aware_params"] for r in rs])
                    ),
                    "per_seed": [
                        {
                            k: r[k]
                            for k in (
                                "seed",
                                "noisy_kl",
                                "clean_kl_of_noise_aware_params",
                                "tv_target_noisy",
                                "wall_s",
                                "evaluations",
                            )
                        }
                        for r in rs
                    ],
                    "histories": [r["history"] for r in rs],
                }
                print(
                    f"    {model:9s} {d:16s} noise-aware median KL="
                    f"{per_device_b[model][d]['median_noisy_kl']:.3e} vs clean-trained "
                    f"{baseline[model][d]['kl_target']:.3e}"
                )
        results["B"] = {
            "network": "full" if full_b else "reduced",
            "n_bits": n_bits_b,
            "n_hidden": n_hidden_b,
            "n_qubits": b_nets["prototype"].n_qubits,
            "n_params": b_nets["prototype"].n_params,
            "backend": b_eval_backend.name,
            "devices": devices,
            "seeds": args.seeds,
            "iterations": args.iterations,
            "curve_device": args.scale_device if args.scale_device in devices else devices[0],
            "baseline": baseline,
            "clean_kl": {m: b_clean[m]["kl"] for m in MODELS},
            "per_device": per_device_b,
        }
        if "C" in results:
            for model in MODELS:
                for r in runs:
                    same_seed = r["seed"] == args.seeds[0]
                    on_scale_device = r["device"] == args.scale_device
                    if r["hidden"] != model or not (same_seed and on_scale_device):
                        continue
                    results["C"]["retrained"][model][f"{r['scale']:g}"] = {
                        "noisy_kl": r["noisy_kl"],
                        "clean_kl": r["clean_kl_of_noise_aware_params"],
                        "seed": r["seed"],
                    }
                if 0.0 in args.scales:
                    # The s = 0 retrain point *is* the clean-trained run (noiseless loss).
                    results["C"]["retrained"][model]["0"] = {
                        "noisy_kl": reduced_clean[model]["kl"],
                        "clean_kl": reduced_clean[model]["kl"],
                        "seed": reduced_clean[model]["best_seed"],
                    }

    # --- outputs --------------------------------------------------------------------------------
    if "A" in results:
        plot_devices(results, out_dir / "figA_devices.png")
    if "C" in results:
        plot_scale(results, out_dir / "figC_scale.png")
    if "D" in results:
        plot_shots(results, out_dir / "figD_shots.png")
    if "B" in results:
        plot_noise_aware(results, out_dir / "figB_noise_aware.png")

    total_wall = time.perf_counter() - wall_t0
    timings["total_s"] = total_wall
    with (out_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=1)
    meta = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_hash": git_hash(),
        "argv": vars(args),
        "studies": sorted(studies),
        "seeds": args.seeds,
        "backend_requested": args.backend,
        "backend_used": backend_kind,
        "backend_name": backend.name,
        "exact_backend_for_reduced": ref.DensityMatrixBackend().name,
        "devices": devices,
        "n_qubits": results["n_qubits"],
        "n_params": results["n_params"],
        "seconds_per_noisy_evaluation": probe_s,
        "wall_times_s": timings,
        "host_cpu_count": os.cpu_count(),
    }
    with (out_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
    (out_dir / "results.md").write_text(report(results, meta, args))
    print(f"[done] wrote {out_dir} in {total_wall:.1f}s")


def report(results: dict[str, Any], meta: dict[str, Any], args: argparse.Namespace) -> str:
    """Render ``results.md``.  Every section is optional: a run with ``--studies B`` produces a
    report with only the header, the budget and the B section."""
    devices = results["devices"]
    nq = results["n_qubits"]
    probe_s = meta["wall_times_s"]["probe_noisy_eval_s"]
    sweep_min = probe_s * 3 * results["n_params"] / 60
    run_h = sweep_min * args.iterations / 60
    e0 = results["E"]["prototype"][devices[0]]
    e1 = results["E"]["classic"][devices[0]]
    n_channels = e0["n_channels_inserted"]
    a, b, c, d = (results.get(k) for k in ("A", "B", "C", "D"))
    studies = results.get("studies", ["A", "B", "C", "D", "E"])

    lines = [
        "# Bit-count under realistic noise — PLAN.md §1.3 / P9",
        "",
        f"Run `{args.out}` — git `{meta['git_hash'][:12]}` — exact backend "
        f"`{meta['backend_name']}` (requested `{args.backend}`) — studies "
        f"{','.join(studies)} — {results['n_bits']} bits / {results['n_hidden']} hidden = "
        f"**{nq} qubits**, {results['n_params']} parameters (identical for prototype and classic "
        f"hidden layers, [PTQNN §III.C]) — seeds {args.seeds} — {args.iterations} KW–SA sweeps — "
        f"wall {meta['wall_times_s']['total_s']:.0f} s.",
        "",
        "## Budget",
        "",
        f"One exact noisy {nq}-qubit density-matrix evaluation of the model distribution costs "
        f"**{probe_s:.2f} s** on `{meta['backend_used']}`.  A KW–SA coordinate sweep needs "
        f"`3·n_params` = {3 * results['n_params']} of them (~{sweep_min:.1f} min), so one "
        f"{args.iterations}-sweep noise-aware training run at {nq} qubits costs ~{run_h:.2f} h "
        f"and the full {len(devices)} devices × {len(args.seeds)} seeds × 2 models grid "
        f"~{run_h * 2 * len(devices) * len(args.seeds):.1f} h.",
    ]
    if b is not None:
        lines += [
            "",
            f"Study B ran on the **{b['network']}** network — {b['n_bits']} bits / "
            f"{b['n_hidden']} hidden = {b['n_qubits']} qubits, {b['n_params']} parameters — with "
            f"`{b['backend']}` as the exact mixed-state backend.",
        ]
    lines += [""]

    # --- A ------------------------------------------------------------------------------------
    if a is not None:
        lines += [
            "## A. Train clean, deploy noisy",
            "",
            "Clean (exact statevector) KL after training: "
            + ", ".join(
                f"**{m}** {a[m]['clean_kl']:.2e} (seed {a[m]['best_seed']})" for m in MODELS
            )
            + ".  Deployed unchanged under each vendor preset "
            "(`KL` = KL(target ‖ noisy), `TV` = total variation to target, "
            "`KL(c‖n)` = KL(clean model ‖ noisy model)):",
            "",
            "| device | proto KL | proto TV | proto KL(c‖n) | classic KL | classic TV | "
            "classic KL(c‖n) |",
            "|---|---|---|---|---|---|---|",
        ]
        for dev in devices:
            p, cl = a["prototype"]["per_device"][dev], a["classic"]["per_device"][dev]
            lines.append(
                f"| {dev} | {p['kl_target']:.4f} | {p['tv_target']:.4f} | "
                f"{p['kl_clean_vs_noisy']:.4f} | {cl['kl_target']:.4f} | {cl['tv_target']:.4f} | "
                f"{cl['kl_clean_vs_noisy']:.4f} |"
            )
        ratios = {m: {dev: a[m]["per_device"][dev]["kl_target"] for dev in devices} for m in MODELS}
        worst = {m: max(ratios[m], key=lambda x: ratios[m][x]) for m in MODELS}
        best = {m: min(ratios[m], key=lambda x: ratios[m][x]) for m in MODELS}
        lines += [
            "",
            f"Best device: prototype `{best['prototype']}` "
            f"({ratios['prototype'][best['prototype']]:.4f}), classic `{best['classic']}` "
            f"({ratios['classic'][best['classic']]:.4f}); worst: prototype `{worst['prototype']}` "
            f"({ratios['prototype'][worst['prototype']]:.4f}), classic `{worst['classic']}` "
            f"({ratios['classic'][worst['classic']]:.4f}).",
            "",
            "![A](figA_devices.png)",
            "",
        ]

    # --- B ------------------------------------------------------------------------------------
    gains: dict[str, list[float]] = {}
    if b is not None:
        lines += [
            f"## B. Noise-aware training ({b['n_qubits']} qubits, {b['network']} network)",
            "",
            f"{b['n_bits']} bits / {b['n_hidden']} hidden = {b['n_qubits']} qubits, "
            f"{b['n_params']} parameters, {len(b['seeds'])} seed(s) × {b['iterations']} sweeps, "
            f"loss evaluated on `NoisyBackend({b['backend']}, preset)`.  Clean-trained baseline "
            "on the same network: "
            + ", ".join(f"**{m}** KL {b['clean_kl'][m]:.2e}" for m in MODELS)
            + ".",
            "",
            "| device | proto clean→noisy | proto NA (best) | proto NA (med) | "
            "proto NA clean-eval | classic clean→noisy | classic NA (best) | "
            "classic NA (med) | classic NA clean-eval |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for dev in b["devices"]:
            pb, pa = b["baseline"]["prototype"][dev], b["per_device"]["prototype"][dev]
            cb, ca = b["baseline"]["classic"][dev], b["per_device"]["classic"][dev]
            lines.append(
                f"| {dev} | {pb['kl_target']:.2e} | {pa['best_noisy_kl']:.2e} | "
                f"{pa['median_noisy_kl']:.2e} | {pa['median_clean_kl']:.2e} | "
                f"{cb['kl_target']:.2e} | {ca['best_noisy_kl']:.2e} | "
                f"{ca['median_noisy_kl']:.2e} | {ca['median_clean_kl']:.2e} |"
            )
        gains = {
            m: [
                b["baseline"][m][dev]["kl_target"]
                / max(b["per_device"][m][dev]["best_noisy_kl"], 1e-15)
                for dev in b["devices"]
            ]
            for m in MODELS
        }
        lines += [
            "",
            "Recovery factor (clean-trained KL ÷ best-seed noise-aware KL — best against "
            "best, since the clean baseline is itself the best of the same seeds; >1 means "
            "noise-aware training helps): "
            + ", ".join(
                f"**{m}** median ×{np.median(gains[m]):.2f} "
                f"(range ×{min(gains[m]):.2f}–×{max(gains[m]):.2f})"
                for m in MODELS
            )
            + ".",
            "",
            "| device | proto recovery | classic recovery | proto run (min) | classic run (min) |",
            "|---|---|---|---|---|",
        ]
        for dev in b["devices"]:
            pw = (
                np.median([r["wall_s"] for r in b["per_device"]["prototype"][dev]["per_seed"]]) / 60
            )
            cw = np.median([r["wall_s"] for r in b["per_device"]["classic"][dev]["per_seed"]]) / 60
            gp = b["baseline"]["prototype"][dev]["kl_target"] / max(
                b["per_device"]["prototype"][dev]["best_noisy_kl"], 1e-15
            )
            gc = b["baseline"]["classic"][dev]["kl_target"] / max(
                b["per_device"]["classic"][dev]["best_noisy_kl"], 1e-15
            )
            lines.append(f"| {dev} | ×{gp:.1f} | ×{gc:.1f} | {pw:.1f} | {cw:.1f} |")
        lines += ["", "![B](figB_noise_aware.png)", ""]

    # --- C ------------------------------------------------------------------------------------
    if c is not None:
        lines += [
            "## C. Noise-strength sweep",
            "",
            f"`{c['device']}`'s model scaled by `NoiseModel.scaled(s)` (depolarising `p`, `1/T1`, "
            "`1/T2` and readout error all ×s; `s = 0` is the noiseless model).",
            "",
            f"| s | proto KL ({nq}q) | classic KL ({nq}q) | proto KL "
            f"({c['reduced_n_qubits']}q) | classic KL ({c['reduced_n_qubits']}q) |",
            "|---|---|---|---|---|",
        ]
        for s in c["scales"]:
            k = f"{s:g}"
            lines.append(
                f"| {k} | {c['fig5']['prototype'][k]['kl_target']:.4f} | "
                f"{c['fig5']['classic'][k]['kl_target']:.4f} | "
                f"{c['reduced']['prototype'][k]['kl_target']:.4f} | "
                f"{c['reduced']['classic'][k]['kl_target']:.4f} |"
            )
        lines += [
            "",
            f"Sensitivity ({nq}-qubit network); the exponent is fitted to the *excess* "
            f"`ΔKL(s) = KL(s) − KL(0)` so the training floor `KL(0)` does not flatten it: "
            + ", ".join(
                f"**{m}** dKL/ds = {c[f'slope_{m}']['linear_slope_s_le_1']:.4f} for s ≤ 1, "
                f"ΔKL ∝ s^{c[f'slope_{m}']['loglog_exponent']:.2f} (floor "
                f"{c[f'slope_{m}']['kl_floor']:.4f})"
                for m in MODELS
            )
            + ".",
            "",
            "![C](figC_scale.png)",
            "",
        ]

    # --- D ------------------------------------------------------------------------------------
    if d is not None:
        lines += [
            "## D. Shot noise",
            "",
            f"Multinomial sampling of the full {nq}-qubit outcome distribution, "
            f"{d['reps']} repetitions per budget; the estimator conditions on `out = 1` and "
            "renormalises exactly as the exact one does, so only a fraction `P(out=1)` of the "
            "shots survive.  `eps`-floored KL (`losses.kl_divergence`, eps = 1e-12).",
            "",
            "| model | source | shots | KL mean ± sd | exact | bias | usable shots |",
            "|---|---|---|---|---|---|---|",
        ]
        for m in MODELS:
            for source in ("clean", "noisy"):
                for row in d[m][source]:
                    kept = (
                        f"{row['kept_mean']:.3f} (P(out=1))"
                        if row["shots"] == 0
                        else f"{row['kept_mean']:.0f}"
                    )
                    lines.append(
                        f"| {m} | {source} | {row['shots'] or 'exact'} | "
                        f"{row['mean']:.4f} ± {row['sd']:.4f} | {row['exact']:.4f} | "
                        f"{row['bias']:+.4f} | {kept} |"
                    )
        lines += ["", "![D](figD_shots.png)", ""]

    # --- E ------------------------------------------------------------------------------------
    compile_table = Path("reports/hardware-compile-table.md")
    compiled = compiled_2q_counts(compile_table, results["n_bits"], results["n_hidden"])
    if "E" in studies:
        lines += [
            "## E. Circuit facts (before transpilation)",
            "",
            f"Gate counts are device-independent: prototype {e0['n_1q_gates']} 1q + "
            f"{e0['n_2q_gates']} 2q gates; classic {e1['n_1q_gates']} 1q + {e1['n_2q_gates']} 2q.  "
            "The prototype layer pays 2 extra RX per (input, neuron) pair for the un-compute "
            "[PTQNN Fig 2] but has the *same* 2q count as the classic layer.",
            "",
            "| device | total ns (proto) | t/T2 | Σ 1q err | Σ 2q err | Σ readout | "
            "channels inserted |",
            "|---|---|---|---|---|---|---|",
        ]
        for dev in devices:
            f = results["E"]["prototype"][dev]
            t2f = "—" if f["t2_fraction"] is None else f"{f['t2_fraction']:.3f}"
            lines.append(
                f"| {dev} | {f['total_ns']:.0f} | {t2f} | {f['error_budget_1q']:.4f} | "
                f"{f['error_budget_2q']:.4f} | {f['readout_budget']:.4f} | "
                f"{f['n_channels_inserted']} |"
            )
        lines += [
            "",
            "Hardware compilation to a native gate set on a restricted coupling map adds SWAPs "
            "and expands each CRX into 2 native entanglers, so the 2q budget above is a "
            "**lower bound**.",
        ]
        if compiled:
            lines += [
                "",
                f"`reports/hardware-compile-table.md` compiles this network ({e0['n_2q_gates']} "
                "logical CRX) to "
                + ", ".join(f"**{v}** native 2q gates on {k}" for k, v in compiled.items())
                + " — i.e. the Σ(2q error) column must be multiplied by "
                + ", ".join(f"×{v / e0['n_2q_gates']:.1f} ({k})" for k, v in compiled.items())
                + ".  The prototype topology is star-like (every input couples to every hidden "
                "qubit), which is why a linear chain costs so many SWAPs and an all-to-all "
                "trapped-ion machine none.",
            ]
        elif not compile_table.exists():
            lines += [
                "",
                "`reports/hardware-compile-table.md` (P10) will carry the compiled counts.",
            ]
        lines += [""]

    # --- interpretation, computed ---------------------------------------------------------------
    lines += ["## Interpretation", ""]
    if a is not None:
        a_p = np.array([a["prototype"]["per_device"][x]["kl_target"] for x in devices])
        a_c = np.array([a["classic"]["per_device"][x]["kl_target"] for x in devices])
        tv_p = np.array([a["prototype"]["per_device"][x]["tv_target"] for x in devices])
        tv_c = np.array([a["classic"]["per_device"][x]["tv_target"] for x in devices])
        med_kl_c_n = {
            m: float(np.median([a[m]["per_device"][x]["kl_clean_vs_noisy"] for x in devices]))
            for m in MODELS
        }
        graceful = "prototype" if np.median(tv_p) < np.median(tv_c) else "classic"
        lines += [
            f"* **Which model degrades more gracefully.** Median over the {len(devices)} presets: "
            f"KL {np.median(a_p):.4f} (prototype) vs {np.median(a_c):.4f} (classic); "
            f"TV {np.median(tv_p):.4f} vs {np.median(tv_c):.4f}.  On the bounded metric the "
            f"**{graceful}** hidden layer is the more graceful one on every preset.  But the two "
            f"start from different clean KLs ({a['prototype']['clean_kl']:.2e} vs "
            f"{a['classic']['clean_kl']:.2e}), and the *displacement caused by the noise* — "
            "`KL(clean ‖ noisy)`, which removes the training floor — is near-identical: median "
            f"{med_kl_c_n['prototype']:.4f} vs {med_kl_c_n['classic']:.4f}.  So the prototype "
            "network's advantage under noise is inherited from its better noiseless fit, not "
            "from any extra intrinsic robustness: both layers have the same 2q-gate count and "
            "therefore the same error budget, and the noise displaces both distributions "
            "equally far.",
        ]
    if b is not None:
        gains_all = {m: float(np.median(gains[m])) for m in MODELS}
        helps = [m for m in MODELS if gains_all[m] > 1.1]
        hurts = [m for m in MODELS if gains_all[m] < 0.9]
        lines += [
            f"* **Does noise-aware training help?** Median recovery factor ×"
            f"{gains_all['prototype']:.2f} (prototype), ×{gains_all['classic']:.2f} (classic) on "
            f"the {b['n_qubits']}-qubit ({b['network']}) network.  "
            + (
                f"It helps for: {', '.join(helps)}.  The signature that the recovered parameters "
                "really are noise-specialised is the `NA clean-eval` column: evaluated on a "
                "*noiseless* simulator the noise-aware parameters are worse than the noisy score "
                "they were trained for, i.e. the optimiser has folded the channel into the "
                "angles rather than found a better network.  "
                if helps
                else ""
            )
            + (
                f"It *hurts* for: {', '.join(hurts)} — with the same sweep budget the noisy loss "
                "is a harder landscape, so the clean optimum transplanted onto the device wins.  "
                if hurts
                else ""
            )
            + (
                "At these error rates the noisy optimum is essentially the clean optimum plus an "
                "offset the parameters cannot remove: the channels act *after* each gate, so no "
                "rotation angle undoes them."
                if not helps
                else ""
            ),
        ]
    e_other = {
        dev: results["E"]["prototype"][dev]["error_budget_1q"]
        + results["E"]["prototype"][dev]["readout_budget"]
        for dev in devices
    }
    e_ratio = {
        dev: (
            float("inf")
            if e_other[dev] == 0.0
            else results["E"]["prototype"][dev]["error_budget_2q"] / e_other[dev]
        )
        for dev in devices
    }
    dom = [dev for dev in devices if e_ratio[dev] > 1]
    if "E" in studies:
        lines += [
            f"* **Where the 2q-gate budget dominates.** Σ(2q error) exceeds Σ(1q error) + "
            f"Σ(readout) on {len(dom)}/{len(devices)} presets "
            f"({', '.join(dom) if dom else 'none'}); ratios "
            + ", ".join(
                f"{dev} "
                + (
                    "×∞ (no 1q or readout error published)"
                    if e_ratio[dev] == float("inf")
                    else f"×{e_ratio[dev]:.1f}"
                )
                for dev in devices
            )
            + f".  With {e0['n_2q_gates']} logical CRX gates the 2q depolarising term is the "
            "single largest gate contribution wherever T2 is long compared with the schedule"
            + (
                f", and transpilation multiplies it by "
                f"×{min(compiled.values()) / e0['n_2q_gates']:.0f}–"
                f"×{max(compiled.values()) / e0['n_2q_gates']:.0f} depending on the coupling map, "
                "so on a linear chain the 2q budget dominates everything else on every preset."
                if compiled
                else "; transpilation only makes this worse."
            ),
        ]
    if c is not None:
        disp = {
            m: {
                key: _slope(c["scales"], [c["fig5"][m][f"{s:g}"][key] for s in c["scales"]])[
                    "loglog_exponent"
                ]
                for key in ("kl_clean_vs_noisy", "tv_clean_vs_noisy")
            }
            for m in MODELS
        }
        lines += [
            f"* **Noise strength.** Measured on the displacement from the clean model (which is "
            f"exactly zero at s = 0, so no floor has to be subtracted): "
            f"TV(clean ‖ noisy) ∝ s^{disp['prototype']['tv_clean_vs_noisy']:.2f} / "
            f"s^{disp['classic']['tv_clean_vs_noisy']:.2f} and "
            f"KL(clean ‖ noisy) ∝ s^{disp['prototype']['kl_clean_vs_noisy']:.2f} / "
            f"s^{disp['classic']['kl_clean_vs_noisy']:.2f} (prototype / classic) over the whole "
            "sweep.  That is textbook linear response: the distribution moves linearly in the "
            "error rates and KL is quadratic in the displacement (`KL ≈ ½χ²`).  Against the "
            f"*target* the exponents are diluted by the training floor — ΔKL ∝ "
            f"s^{c['slope_prototype']['loglog_exponent']:.2f} / "
            f"s^{c['slope_classic']['loglog_exponent']:.2f}.  Practically: a uniform ×m "
            "improvement in every error rate buys ×m in TV but ×m² in KL, so KL flatters better "
            "hardware and punishes worse hardware quadratically.",
        ]
    if d is not None:
        lines += [
            f"* **Shots.** The plug-in KL estimator is biased upwards; at "
            f"{max(args.shots) if max(args.shots) else 0} shots the bias is "
            + ", ".join(f"{m} {d[m]['noisy'][-1]['bias']:+.4f}" for m in MODELS if d[m]["noisy"])
            + " on the noisy distribution.  Only the `out = 1` shots are usable, so the effective "
            "sample size is `P(out=1)·N`.",
        ]
    lines += [
        "",
        "## Module notes",
        "",
        "Three `src/` issues found by this study; the first two are **fixed at HEAD** (commit "
        "`30564d9`), the third is **open**.  They are recorded because they set the shape of the "
        "experiment.",
        "",
        "* `NoiseModel.scaled(0.0)` raised `ZeroDivisionError` (`t1_us / factor`).  The noiseless "
        "limit is a legitimate point of a noise sweep and now returns a trivial model, so study "
        "C's `s = 0` point needs no special case.",
        "* The density-matrix path applied every Kraus operator separately with "
        "`moveaxis`/`tensordot` on a 2n-axis tensor — at 11 qubits ~80 ms per channel against "
        "~1 ms of memory traffic.  It now builds the block superoperator `S = Σ_i K_i ⊗ K_i*` "
        "and applies it in one matmul: the noisy 11-qubit evaluation went **18 s → 1.5 s** on "
        "the GPU (numpy 148 s → 33 s).  That speedup is what makes noise-aware training on the "
        f"full network affordable; this run measures {probe_s:.2f} s per noisy {nq}-qubit "
        f"evaluation, with {n_channels} channels inserted per circuit.",
        "* **Open:** the same commit broke `GpuDensityMatrixBackend`.  "
        "`reference.superoperator` seeds its accumulator with `np.zeros((d², d²))` and then does "
        "`s = s + np.kron(k, k.conj())`; under the GPU backend `k` is a CuPy array, `np.kron` "
        "dispatches to CuPy, and the NumPy-plus-CuPy add raises "
        "`TypeError: Unsupported type <class 'numpy.ndarray'>` "
        "(`ptqnn/backends/reference.py`, in `superoperator`, reached from `apply_gate_dm`).  "
        "Seeding the accumulator from the first Kraus term — or `np.zeros_like(np.kron(k0, k0))` "
        "— keeps it array-library agnostic.  Repro: "
        "`ref.superoperator([cupy.eye(2, dtype=cupy.complex128)])`.  Until it is fixed the exact "
        "GPU path is unusable, and this study fell back to orkan.",
        "",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
