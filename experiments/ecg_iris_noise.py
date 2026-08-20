"""ECG denoising and Iris classification under vendor-calibrated CP-map noise.

Deploys the *already trained* parameters of two reproduction runs on noisy backends — nothing is
retrained except in study C.  All physics lives in ``ptqnn.noise``, ``ptqnn.tasks`` and
``ptqnn.backends``; this script drives them, records and plots.

* **A. ECG (7–9 qubits, exact density matrix).**  For every cell of ``[PTQNN Table I]``
  (CQNN/PTQNN × 1/2/3 hidden neurons) the denoised test signal is recomputed with
  ``NoisyBackend(numpy density matrix, preset)`` for every vendor preset, then for the
  ``ibm_fez`` noise model scaled by ``--scales`` ``[NoiseModel.scaled]``, and finally under shot
  noise — ``ŝ(t) = Binomial(N, P(out = 1|window t))/N``, ``--shot-reps`` repetitions per budget.
  Metrics are the paper's ``[PTQNN Eq 6–8]``: PSNR (with ``R`` the peak-to-peak range of the
  *whole* clean signal, as in ``ecg_table1.py``) and the sliding-window SSIM.
* **B. Iris (17 qubits, Monte-Carlo trajectories).**  The trained prototype and classic
  parameters are evaluated under every preset with ``TrajectoryBackend``/``GpuTrajectoryBackend``:
  test accuracy and KL(target ‖ model) on the test superposition, with ``--traj-reps`` different
  trajectory seeds as the error bar, plus the ``ibm_fez`` scaled sweep.
  ``[ASSUMPTION]`` **State preparation is assumed perfect**: the dataset superposition enters as
  ``Circuit.initial_state``, so only the network's own gates carry noise.  A real device would
  have to synthesise that 20-term superposition with a state-preparation circuit whose own error
  is not modelled here, and the numbers below are therefore optimistic.
* **C. Noise-aware fine-tuning (ECG, 7 qubits, 1 prototype neuron, ``--ft-device``).**
  ``--ft-sweeps`` KW–SA coordinate sweeps starting from the trained parameters, with the loss
  evaluated on the noisy backend; PSNR before/after, scored both noisily and cleanly.

Budget.  A 9-qubit noisy density-matrix evaluation costs ~0.8 s of CPU per sliding window and
the test block holds ~900 of them, so the full grid is several CPU-hours; the script therefore
*probes the machine first* and picks the number of test windows that fits ``--ecg-budget-s``
(override with ``--ecg-windows``).  The evaluated block is contiguous and centred in the test
segment, and every comparison — clean simulator, each device, each scale — uses the same block.
Iris is likewise probed and ``--trajectories`` is reduced to fit ``--iris-budget-s``.  Study B
runs concurrently with A in a ``spawn`` process, because A is CPU-bound and B is GPU-bound.

    export LD_LIBRARY_PATH=/usr/lib/wsl/lib     # CuPy needs this *before* the process starts
    uv run python experiments/ecg_iris_noise.py --devices all --scales 0,0.5,1,2,4 \
        --shots 0,1000,10000 --trajectories 2000
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
from datetime import UTC, datetime
from functools import cache
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
from ptqnn.noise.model import NoiseModel, NoisyBackend
from ptqnn.tasks.ecg import model as ecg_model
from ptqnn.tasks.ecg import train as ecg_train
from ptqnn.train import kwsa, losses

Real = npt.NDArray[np.float64]

ROOT = Path(__file__).resolve().parents[1]
KINDS: tuple[HiddenKind, HiddenKind] = ("classic", "prototype")
COLOURS = {"prototype": "#4c72b0", "classic": "#c44e52"}
PSNR_TARGET_DB = 35.0
"""Reporting threshold: ``[PTQNN Table I]``'s two- and three-neuron cells sit at 38–40 dB, and
35 dB is ~3 dB below them — the level at which a device still delivers a *useful* denoiser
rather than merely passing the input through (the noisy input itself scores 32.0 dB)."""


# =================================================================================================
# pure helpers (unit-tested in tests/test_ecg_iris_noise_script.py)
# =================================================================================================


def parse_list(s: str) -> list[str]:
    return [v for v in s.replace(",", " ").split() if v]


def parse_ints(s: str) -> list[int]:
    return [int(v) for v in parse_list(s)]


def parse_floats(s: str) -> list[float]:
    return [float(v) for v in parse_list(s)]


def centred_block(indices: npt.ArrayLike, n: int) -> npt.NDArray[np.int64]:
    """The ``n`` consecutive entries in the middle of ``indices``.

    The metrics need a *contiguous* stretch of samples (``ssim_windowed`` slides a 7-sample
    window along the signal), so a sub-sampled evaluation must take a block, not a scatter.  The
    middle of the ECG test segment is where the second beat of ``[PTQNN Fig 13]`` sits, so a
    centred block always contains an R peak rather than only baseline.
    """
    idx = np.asarray(indices, dtype=np.int64).ravel()
    if n <= 0 or n >= idx.size:
        return idx
    start = (idx.size - n) // 2
    return idx[start : start + n]


def cost_units(n_hidden_values: list[int], reference_n_hidden: int = 3) -> float:
    """Cost of one window through *all* cells, in units of one ``reference`` evaluation.

    An ``n``-qubit density-matrix step is ``O(4^n)`` and a cell with ``k`` hidden neurons has
    ``n = 5 + k + 1`` qubits, so cell ``k`` costs ``4^(k − reference)`` reference evaluations.
    Used to turn a single timing probe into a window budget.
    """
    return float(sum(4.0 ** (k - reference_n_hidden) for k in n_hidden_values))


def budget_windows(
    seconds: float, per_reference_eval_s: float, combos: int, per_window_units: float
) -> int:
    """How many windows fit in ``seconds`` — ``combos`` × ``per_window_units`` per window."""
    if per_reference_eval_s <= 0 or combos <= 0 or per_window_units <= 0:
        raise ValueError("probe timing, combos and cost units must be positive")
    return max(1, int(seconds / (per_reference_eval_s * combos * per_window_units)))


def shot_activations(p: npt.ArrayLike, shots: int, rng: np.random.Generator) -> Real:
    """``Binomial(shots, p)/shots`` — the finite-shot estimate of ``P(out = 1)`` per window.

    The ECG read-out is a single-qubit activation ``[PTQNN Fig 12]``, so the shot statistics are
    binomial per window rather than multinomial over the whole register: ``shots`` repetitions
    of the circuit, count the ``out = 1`` outcomes.  Unbiased, with variance ``p(1−p)/shots``.
    """
    if shots <= 0:
        raise ValueError("shots must be positive")
    q = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    return np.asarray(rng.binomial(shots, q) / shots, dtype=np.float64)


def shots_for_db_loss(mse: float, mean_variance: float, db: float) -> float:
    """Shots per window needed to keep the PSNR loss below ``db``.

    Shot noise is independent of the reconstruction error, so the sampled estimator has
    ``MSE(N) = MSE_∞ + E_t[p_t(1−p_t)]/N`` and
    ``ΔPSNR = 10 log10(1 + E[p(1−p)]/(N·MSE_∞))``.  Setting that to ``db`` gives
    ``N = E[p(1−p)] / (MSE_∞ (10^{db/10} − 1))``.  Note the answer scales with ``1/MSE_∞``: the
    *better* the denoiser, the more shots it needs to show its quality.
    """
    if mse <= 0 or db <= 0:
        raise ValueError("mse and db must be positive")
    return float(mean_variance / (mse * (10.0 ** (db / 10.0) - 1.0)))


def devices_above(rows: dict[str, float], threshold: float) -> list[str]:
    """Devices whose metric clears ``threshold``, best first."""
    return [d for d, v in sorted(rows.items(), key=lambda kv: -kv[1]) if v >= threshold]


def loglog_exponent(scales: list[float], values: list[float], floor: float) -> float:
    """Exponent ``k`` of ``value − floor ∝ s^k``; ``nan`` if fewer than two usable points.

    As in ``experiments/bitcount_noise.py``: the *excess* over the noiseless value is the
    noise-induced part, and its log–log slope says whether the response is linear in the error
    rates (``k ≈ 1``) or faster.
    """
    s = np.asarray(scales, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64) - float(floor)
    ok = (s > 0) & (v > 0)
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(s[ok]), np.log(v[ok]), 1)[0])


def iris_scores(
    joint: npt.ArrayLike,
    codes: npt.ArrayLike,
    labels: npt.ArrayLike,
    target: npt.ArrayLike,
    levels_from: npt.ArrayLike | None = None,
) -> dict[str, float]:
    """KL, accuracy and input-register leak from one ``P(inputs = x, out = 1)`` vector.

    Both observables of ``[PTQNN §III.C]`` come from the *same* backend call: the KL of
    ``ptqnn.tasks.iris.model.kl_loss`` uses the renormalised joint, the accuracy of
    ``model.accuracy`` uses the same joint post-selected onto the dataset codes.  Splitting them
    into two backend calls would double the trajectory budget for no extra information; the
    equivalence is pinned in ``tests/test_ecg_iris_noise_script.py``.
    """
    from ptqnn.tasks.iris import model as iris_model

    j = np.asarray(joint, dtype=np.float64)
    c = np.asarray(codes, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int64)
    uniq = np.unique(c)
    z = j.sum()
    full = j / z if z > 0 else np.full(j.shape, 1.0 / j.size)
    restricted = np.zeros_like(j)
    restricted[uniq] = j[uniq]
    zr = restricted.sum()
    restricted = restricted / zr if zr > 0 else np.full(j.shape, 1.0 / j.size)
    values = restricted[uniq]
    scale = None if levels_from is None else iris_model.fit_level_scale(np.asarray(levels_from))
    per_code = iris_model.labels_from_values(values, scale)
    pred = per_code[np.searchsorted(uniq, c)]
    return {
        "kl": losses.kl_divergence(np.asarray(target, dtype=np.float64), full),
        "accuracy": float(np.mean(pred == y)),
        "leak": float(1.0 - full[uniq].sum()),
    }


# =================================================================================================
# shared plumbing
# =================================================================================================


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


@cache
def noise_model(device: str, scale: float) -> NoiseModel:
    return presets.device(device).noise_model().scaled(scale)


def noisy_backend(device: str, scale: float, inner: Backend | None = None) -> Backend:
    nm = noise_model(device, scale)
    if nm.is_trivial():
        return ref.StatevectorBackend()
    return NoisyBackend(inner or ref.DensityMatrixBackend(), nm)


# =================================================================================================
# A + C: ECG
# =================================================================================================

_ECG: dict[str, Any] = {}
"""Read-only worker state, populated before the fork pool is created."""

_PROBE: dict[str, Any] = {}
"""Timing-probe state; separate from :data:`_ECG` because the probe runs *before* the window
budget — and hence the evaluated block — is known."""


def _probe_job(i: int) -> float:
    """One noisy window evaluation, timed inside the worker."""
    t0 = time.perf_counter()
    ecg_model.predict(
        _PROBE["net"],
        _PROBE["params"],
        _PROBE["X"][i : i + 1],
        noisy_backend(_PROBE["device"], 1.0),
        _PROBE["encoding"],
    )
    return time.perf_counter() - t0


def _ecg_net(cell: dict[str, Any]) -> Network:
    return Network(n_inputs=5, n_hidden=cell["n_hidden"], hidden=cell["kind"])


def _ecg_job(job: tuple[int, str, float, int, int]) -> tuple[int, str, float, int, list[float]]:
    """Denoise windows ``[i0, i1)`` of the evaluation block with one (device, scale)."""
    cell_idx, device, scale, i0, i1 = job
    cell = _ECG["cells"][cell_idx]
    net = _ecg_net(cell)
    yhat = ecg_model.predict(
        net,
        np.asarray(cell["params"], dtype=np.float64),
        _ECG["X"][i0:i1],
        noisy_backend(device, scale),
        _ECG["encoding"],
    )
    return cell_idx, device, scale, i0, [float(v) for v in yhat]


def ecg_metrics(y: Real, yhat: Real, clean_hat: Real, R: float) -> dict[str, float]:
    """``[PTQNN Eq 6–8]`` against the ground truth, plus the displacement from the ideal model."""
    return {
        "psnr": ecg_model.psnr(y, yhat, R),
        "ssim_windowed": ecg_model.ssim_windowed(y, yhat),
        "ssim": ecg_model.ssim(y, yhat),
        "mse": ecg_model.mse(y, yhat),
        "psnr_vs_clean_model": ecg_model.psnr(clean_hat, yhat, R),
        "mse_vs_clean_model": ecg_model.mse(clean_hat, yhat),
    }


def shot_study(
    p: Real, y: Real, R: float, shots_list: list[int], reps: int, seed: int
) -> dict[str, Any]:
    """PSNR of the binomially sampled activations, mean ± sd over ``reps`` repetitions."""
    rng = np.random.default_rng(seed)
    exact = ecg_model.psnr(y, p, R)
    mse_inf = ecg_model.mse(y, p)
    var = float(np.mean(p * (1.0 - p)))
    rows: list[dict[str, Any]] = []
    for shots in shots_list:
        if shots <= 0:
            rows.append(
                {"shots": 0, "psnr_mean": exact, "psnr_sd": 0.0, "ssim_windowed_mean": None}
            )
            continue
        ps, ss = [], []
        for _ in range(reps):
            s = shot_activations(p, shots, rng)
            ps.append(ecg_model.psnr(y, s, R))
            ss.append(ecg_model.ssim_windowed(y, s))
        rows.append(
            {
                "shots": shots,
                "psnr_mean": float(np.mean(ps)),
                "psnr_sd": float(np.std(ps, ddof=1)) if reps > 1 else 0.0,
                "psnr_predicted": float(10 * np.log10(R**2 / (mse_inf + var / shots))),
                "ssim_windowed_mean": float(np.mean(ss)),
                "psnr_values": [float(v) for v in ps],
            }
        )
    return {
        "exact_psnr": exact,
        "mse_inf": mse_inf,
        "mean_variance": var,
        "shots_for_1db": shots_for_db_loss(mse_inf, var, 1.0),
        "rows": rows,
    }


def _ft_chunk(payload: tuple[list[float], int, int]) -> tuple[float, int]:
    """Squared error of the noisy model on training windows ``[i0, i1)``."""
    params, i0, i1 = payload
    net = _ECG["ft_net"]
    yhat = ecg_model.predict(
        net,
        np.asarray(params, dtype=np.float64),
        _ECG["ft_X"][i0:i1],
        noisy_backend(_ECG["ft_device"], 1.0),
        _ECG["encoding"],
    )
    return float(np.sum((yhat - _ECG["ft_y"][i0:i1]) ** 2)), int(i1 - i0)


# =================================================================================================
# B: Iris (separate process — GPU, concurrent with A)
# =================================================================================================


def iris_worker(job: dict[str, Any], out_path: str) -> None:
    """One full Iris study; writes JSON so the parent need not unpickle numpy from a spawn pipe."""
    import numpy as _np

    from ptqnn.backends import reference as _ref
    from ptqnn.tasks.iris import data as iris_data
    from ptqnn.tasks.iris import model as iris_model
    from ptqnn.tasks.iris import train as iris_train

    t_start = time.perf_counter()
    split = iris_data.load_split(seed=job["seed"], train_size=job["train_size"])
    cfg = iris_train.IrisConfig(**job["config"])
    nets = {k: iris_train.network(cfg, k) for k in KINDS}
    params = {k: _np.asarray(job["params"][k], dtype=_np.float64) for k in KINDS}
    n_qubits = nets["prototype"].n_qubits

    codes, labels = split.test_codes, split.test_labels
    target = iris_model.target_distribution(codes, labels)
    psi0 = {
        k: iris_model.dataset_superposition(codes, n_qubits, nets[k].input_qubits) for k in KINDS
    }

    def make_backend(n_traj: int, seed: int) -> tuple[Any, str]:
        if job["backend"] == "gpu":
            from ptqnn.backends.gpu import GpuTrajectoryBackend

            be = GpuTrajectoryBackend(
                n_trajectories=n_traj, seed=seed, batch_size=min(job["batch"], n_traj)
            )
            return be, be.name
        from ptqnn.backends.trajectories import TrajectoryBackend

        be = TrajectoryBackend(
            n_trajectories=n_traj, seed=seed, batch_size=min(job["batch"], n_traj)
        )
        return be, be.name

    def evaluate(kind: str, device: str, scale: float, n_traj: int, seed: int) -> dict[str, Any]:
        net = nets[kind]
        nm = presets.device(device).noise_model().scaled(scale)
        if nm.is_trivial():
            inner: Any = _ref.StatevectorBackend()
            be: Any = inner
            name = inner.name
        else:
            inner, name = make_backend(n_traj, seed)
            be = NoisyBackend(inner, nm)
        t0 = time.perf_counter()
        p = be.probabilities(net.circuit(params[kind], initial_state=psi0[kind]))
        joint = _ref.marginal(p, [*net.input_qubits, net.output_qubit]).reshape(2**net.n_inputs, 2)[
            :, 1
        ]
        sc = iris_scores(joint, codes, labels, target)
        return {**sc, "seconds": time.perf_counter() - t0, "backend": name, "trajectories": n_traj}

    # --- clean reference (exact, free) -----------------------------------------------------------
    clean = {}
    for kind in KINDS:
        net = nets[kind]
        p = _ref.StatevectorBackend().probabilities(
            net.circuit(params[kind], initial_state=psi0[kind])
        )
        joint = _ref.marginal(p, [*net.input_qubits, net.output_qubit]).reshape(2**net.n_inputs, 2)[
            :, 1
        ]
        clean[kind] = iris_scores(joint, codes, labels, target)

    # --- probe: one noisy evaluation, to size the trajectory budget ------------------------------
    probe_n = job["probe_trajectories"]
    t0 = time.perf_counter()
    evaluate("prototype", job["devices"][0], 1.0, probe_n, 0)
    probe_s = time.perf_counter() - t0
    per_traj = probe_s / probe_n
    n_evals = (
        len(job["devices"]) * len(KINDS) * job["reps"]
        + len(job["scales_extra"]) * len(KINDS) * job["reps"]
    )
    affordable = int(job["budget_s"] / max(per_traj * n_evals, 1e-9))
    n_traj = int(min(job["trajectories"], max(job["min_trajectories"], affordable)))
    n_traj = max(1, n_traj - n_traj % 8)

    results: dict[str, Any] = {
        "n_qubits": n_qubits,
        "clean": clean,
        "split": split.summary(),
        "probe": {
            "trajectories": probe_n,
            "seconds": probe_s,
            "s_per_trajectory": per_traj,
            "n_evaluations_planned": n_evals,
        },
        "trajectories": n_traj,
        "requested_trajectories": job["trajectories"],
        "reduced": n_traj < job["trajectories"],
        "reps": job["reps"],
        "devices": job["devices"],
        "per_device": {k: {} for k in KINDS},
        "scale": {k: {} for k in KINDS},
        "state_prep": "perfect (Circuit.initial_state) — [ASSUMPTION], see module docstring",
    }
    print(
        f"[B] probe {probe_s:.1f} s for {probe_n} trajectories ({per_traj:.3f} s/traj); "
        f"{n_evals} evaluations planned → {n_traj} trajectories each "
        f"(requested {job['trajectories']})",
        flush=True,
    )

    for kind in KINDS:
        for device in job["devices"]:
            reps = [
                evaluate(kind, device, 1.0, n_traj, 1000 + r * 7919) for r in range(job["reps"])
            ]
            results["per_device"][kind][device] = _aggregate(reps)
            r = results["per_device"][kind][device]
            print(
                f"[B] {kind:9s} {device:16s} acc {r['accuracy_mean']:.3f}±{r['accuracy_sd']:.3f} "
                f"KL {r['kl_mean']:.4f}±{r['kl_sd']:.4f} ({r['seconds']:.0f}s)",
                flush=True,
            )
        for s in job["scales_extra"]:
            reps = [
                evaluate(kind, job["scale_device"], s, n_traj, 2000 + r * 7919)
                for r in range(job["reps"])
            ]
            results["scale"][kind][f"{s:g}"] = _aggregate(reps)
        results["scale"][kind]["1"] = results["per_device"][kind][job["scale_device"]]
        results["scale"][kind]["0"] = {
            "accuracy_mean": clean[kind]["accuracy"],
            "accuracy_sd": 0.0,
            "kl_mean": clean[kind]["kl"],
            "kl_sd": 0.0,
            "leak_mean": clean[kind]["leak"],
            "seconds": 0.0,
            "reps": [],
        }
    results["wall_seconds"] = time.perf_counter() - t_start
    Path(out_path).write_text(json.dumps(results, indent=2) + "\n")


def _aggregate(reps: list[dict[str, Any]]) -> dict[str, Any]:
    def stat(key: str) -> tuple[float, float]:
        a = np.asarray([r[key] for r in reps], dtype=np.float64)
        return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0

    acc_m, acc_s = stat("accuracy")
    kl_m, kl_s = stat("kl")
    leak_m, _ = stat("leak")
    return {
        "accuracy_mean": acc_m,
        "accuracy_sd": acc_s,
        "kl_mean": kl_m,
        "kl_sd": kl_s,
        "leak_mean": leak_m,
        "seconds": float(sum(r["seconds"] for r in reps)),
        "reps": reps,
    }


# =================================================================================================
# figures
# =================================================================================================


def fig_devices(res: dict[str, Any], path: Path) -> None:
    a = res["A"]
    devices = a["devices"]
    cells = a["cells"]
    x = np.arange(len(devices))
    fig, (ax_p, ax_s) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    width = 0.8 / len(cells)
    for i, cell in enumerate(cells):
        key = cell["key"]
        off = (i - (len(cells) - 1) / 2) * width
        psnr = [a["per_device"][key][d]["psnr"] for d in devices]
        ssim = [a["per_device"][key][d]["ssim_windowed"] for d in devices]
        label = f"{cell['model']} {cell['n_hidden']}"
        colour = COLOURS[cell["kind"]]
        alpha = 0.4 + 0.3 * (cell["n_hidden"] - 1)
        ax_p.bar(
            x + off, psnr, width, color=colour, alpha=alpha, edgecolor="black", lw=0.3, label=label
        )
        ax_s.bar(
            x + off, ssim, width, color=colour, alpha=alpha, edgecolor="black", lw=0.3, label=label
        )
        ax_p.axhline(a["clean"][key]["psnr"], color=colour, ls=":", lw=0.7, alpha=alpha)
    ax_p.axhline(PSNR_TARGET_DB, color="black", ls="--", lw=1.2, label=f"{PSNR_TARGET_DB:g} dB")
    ax_p.axhline(a["input_psnr"], color="grey", ls="-.", lw=1.2, label="noisy input (no denoising)")
    ax_p.set_ylabel("PSNR [dB]")
    ax_p.set_ylim(bottom=min(25.0, a["input_psnr"] - 3))
    ax_p.set_title(
        f"A. ECG denoising under vendor noise — {a['n_windows']} test windows "
        f"(samples {a['block'][0]}–{a['block'][-1]}), exact density matrix, σ={a['sigma']}"
    )
    ax_p.legend(fontsize=7, ncol=4)
    ax_s.set_ylabel("SSIM (sliding window)")
    ax_s.set_xticks(x)
    ax_s.set_xticklabels(devices, rotation=20, ha="right")
    ax_s.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_scale_and_shots(res: dict[str, Any], path: Path) -> None:
    a = res["A"]
    fig, (ax_s, ax_n) = plt.subplots(1, 2, figsize=(13, 5))
    scales = a["scales"]
    for cell in a["cells"]:
        key = cell["key"]
        ys = [a["scale"][key][f"{s:g}"]["psnr"] for s in scales]
        ax_s.plot(
            scales,
            ys,
            "o-",
            color=COLOURS[cell["kind"]],
            alpha=0.4 + 0.3 * (cell["n_hidden"] - 1),
            lw=1.6,
            label=f"{cell['model']} {cell['n_hidden']}",
        )
    ax_s.axhline(PSNR_TARGET_DB, color="black", ls="--", lw=1)
    ax_s.set_xlabel(f"noise scale s  (NoiseModel.scaled(s) of {a['scale_device']})")
    ax_s.set_ylabel("PSNR [dB]")
    ax_s.set_title("A. PSNR vs noise strength")
    ax_s.grid(alpha=0.25)
    ax_s.legend(fontsize=7, ncol=2)

    for cell in a["cells"]:
        key = cell["key"]
        for source, style in (("clean", "o-"), ("noisy", "s--")):
            rows = [r for r in a["shots"][key][source]["rows"] if r["shots"] > 0]
            if not rows:
                continue
            ax_n.errorbar(
                [r["shots"] for r in rows],
                [r["psnr_mean"] for r in rows],
                yerr=[r["psnr_sd"] for r in rows],
                fmt=style,
                capsize=3,
                lw=1.3,
                color=COLOURS[cell["kind"]],
                alpha=0.4 + 0.3 * (cell["n_hidden"] - 1),
                label=f"{cell['model']} {cell['n_hidden']} ({source})",
            )
        ax_n.axhline(
            a["shots"][key]["clean"]["exact_psnr"],
            color=COLOURS[cell["kind"]],
            ls=":",
            lw=0.7,
            alpha=0.5,
        )
    ax_n.set_xscale("log")
    ax_n.set_xlabel("shots per window")
    ax_n.set_ylabel("PSNR [dB]")
    ax_n.axhline(PSNR_TARGET_DB, color="black", ls="--", lw=1)
    ax_n.set_title(f"A. PSNR vs shot budget ({a['shot_reps']} repetitions)")
    ax_n.grid(alpha=0.25)
    ax_n.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_iris(res: dict[str, Any], path: Path) -> None:
    b = res.get("B")
    if not b:
        return
    devices = b["devices"]
    x = np.arange(len(devices))
    fig, (ax_a, ax_k) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    width = 0.38
    for i, kind in enumerate(KINDS):
        off = (i - 0.5) * width
        acc = [b["per_device"][kind][d]["accuracy_mean"] for d in devices]
        acc_e = [b["per_device"][kind][d]["accuracy_sd"] for d in devices]
        kl = [b["per_device"][kind][d]["kl_mean"] for d in devices]
        kl_e = [b["per_device"][kind][d]["kl_sd"] for d in devices]
        ax_a.bar(
            x + off,
            acc,
            width,
            yerr=acc_e,
            capsize=3,
            color=COLOURS[kind],
            edgecolor="black",
            lw=0.3,
            label=kind,
        )
        ax_k.bar(
            x + off,
            kl,
            width,
            yerr=kl_e,
            capsize=3,
            color=COLOURS[kind],
            edgecolor="black",
            lw=0.3,
            label=kind,
        )
        ax_a.axhline(
            b["clean"][kind]["accuracy"], color=COLOURS[kind], ls="--", lw=1, label=f"{kind} clean"
        )
        ax_k.axhline(b["clean"][kind]["kl"], color=COLOURS[kind], ls="--", lw=1)
    ax_a.set_ylabel("test accuracy")
    ax_a.set_ylim(0, 1.05)
    ax_a.set_title(
        f"B. Iris ({b['n_qubits']} qubits) under vendor noise — {b['trajectories']} "
        f"trajectories × {b['reps']} seeds, perfect state preparation [ASSUMPTION]"
    )
    ax_a.legend(fontsize=8, ncol=2)
    ax_k.set_yscale("log")
    ax_k.set_ylabel("KL(target ‖ model), test")
    ax_k.set_xticks(x)
    ax_k.set_xticklabels(devices, rotation=20, ha="right")
    ax_k.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_finetune(res: dict[str, Any], path: Path) -> None:
    c = res.get("C")
    if not c:
        return
    fig, (ax, ax_h) = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = ["noisy eval", "clean eval"]
    before = [c["before"]["noisy"]["psnr"], c["before"]["clean"]["psnr"]]
    after = [c["after"]["noisy"]["psnr"], c["after"]["clean"]["psnr"]]
    x = np.arange(2)
    ax.bar(
        x - 0.19, before, 0.38, label="trained clean", color="#c44e52", edgecolor="black", lw=0.3
    )
    ax.bar(
        x + 0.19,
        after,
        0.38,
        label=f"+{c['sweeps']} noisy KW–SA sweeps",
        color="#4c72b0",
        edgecolor="black",
        lw=0.3,
    )
    ax.axhline(PSNR_TARGET_DB, color="black", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("PSNR [dB]")
    ax.set_ylim(bottom=min(before + after) - 2)
    ax.set_title(f"C. noise-aware fine-tuning ({c['device']}, 7 qubits, 1 prototype)")
    ax.legend(fontsize=8)
    ax_h.plot(c["history"], "o-", color="#4c72b0")
    ax_h.set_xlabel("KW–SA sweep")
    ax_h.set_ylabel("noisy training MSE")
    ax_h.set_title("fine-tuning loss")
    ax_h.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =================================================================================================
# report
# =================================================================================================


def _shot_cell(row: dict[str, Any] | None) -> str:
    return "—" if row is None else f"{row['psnr_mean']:.2f}±{row['psnr_sd']:.2f}"


def write_report(res: dict[str, Any], meta: dict[str, Any], path: Path) -> None:
    a = res["A"]
    b = res.get("B")
    c = res.get("C")
    L: list[str] = []
    L.append("# ECG and Iris under vendor-calibrated noise\n")
    L.append(
        f"Run `{meta['out']}` — git `{meta['git_hash'][:12]}` — wall {meta['wall_seconds']:.0f} s"
        f" — ECG backend `{a['backend']}`"
        + (f", Iris backend `{b['backend_kind']}`" if b else "")
        + ".  Trained parameters read from "
        f"`{meta['ecg_run']}` (ECG) and `{meta['iris_run']}` (Iris); **nothing is retrained "
        "except study C**.\n"
    )
    L.append("## Budget and what was actually evaluated\n")
    L.append(
        f"One 9-qubit noisy density-matrix window evaluation costs **{a['probe_s']:.3f} s** of "
        f"wall time at {meta['jobs']}-way parallelism on this (shared) machine.  The full test "
        f"block holds {a['n_test_windows']} windows and the grid has "
        f"{a['n_combos']} (device, scale) combinations × 6 cells, so the whole test segment "
        f"would cost ≈{a['full_estimate_s'] / 60:.0f} min; the script evaluated the "
        f"**centred contiguous block of {a['n_windows']} windows** "
        f"(samples {a['block'][0]}–{a['block'][-1]} of {a['n_samples']}), which contains the "
        "second beat of [PTQNN Fig 13].  Every row below — clean, each device, each scale — is "
        "computed on that same block, so the comparisons are like-for-like; the absolute PSNR "
        "differs from `ecg_table1.py`'s full-block numbers, which are quoted as `paper block` "
        "for reference.\n"
    )
    L.append("## A. ECG denoising, clean-trained parameters deployed under vendor noise\n")
    L.append(
        "PSNR / SSIM(sliding) against the clean ground truth; `Δ` is the PSNR loss relative to "
        "the same parameters on a noiseless simulator, on the same block.\n"
    )
    head = "| device | " + " | ".join(f"{c['model']} {c['n_hidden']}" for c in a["cells"]) + " |"
    L.append(head)
    L.append("|---" * (len(a["cells"]) + 1) + "|")
    L.append(
        "| *clean simulator* | "
        + " | ".join(f"{a['clean'][c['key']]['psnr']:.2f}" for c in a["cells"])
        + " |"
    )
    for d in a["devices"]:
        cells = " | ".join(
            f"{a['per_device'][c['key']][d]['psnr']:.2f} "
            f"(Δ{a['per_device'][c['key']][d]['psnr'] - a['clean'][c['key']]['psnr']:+.2f})"
            for c in a["cells"]
        )
        L.append(f"| {d} | {cells} |")
    L.append("")
    L.append("SSIM (sliding window), same layout:\n")
    L.append(head)
    L.append("|---" * (len(a["cells"]) + 1) + "|")
    L.append(
        "| *clean simulator* | "
        + " | ".join(f"{a['clean'][c['key']]['ssim_windowed']:.4f}" for c in a["cells"])
        + " |"
    )
    for d in a["devices"]:
        L.append(
            f"| {d} | "
            + " | ".join(f"{a['per_device'][c['key']][d]['ssim_windowed']:.4f}" for c in a["cells"])
            + " |"
        )
    L.append("")
    L.append(
        f"Noisy input (no denoising at all) on this block: **{a['input_psnr']:.2f} dB**, "
        f"SSIM(sliding) {a['input_ssim_windowed']:.4f}.\n"
    )
    for c_ in a["cells"]:
        keep = devices_above(
            {d: a["per_device"][c_["key"]][d]["psnr"] for d in a["devices"]}, PSNR_TARGET_DB
        )
        L.append(
            f"* **{c_['model']} {c_['n_hidden']}** (clean {a['clean'][c_['key']]['psnr']:.2f} dB): "
            + (
                f"≥{PSNR_TARGET_DB:g} dB on {len(keep)}/{len(a['devices'])} devices — "
                + ", ".join(keep)
                if keep
                else f"no device reaches {PSNR_TARGET_DB:g} dB"
            )
        )
    L.append("")
    L.append("![A](figA_ecg_devices.png)\n")

    L.append(f"## A2. Noise-strength sweep — `{a['scale_device']}` scaled\n")
    L.append("| s | " + " | ".join(f"{c['model']} {c['n_hidden']}" for c in a["cells"]) + " |")
    L.append("|---" * (len(a["cells"]) + 1) + "|")
    for s in a["scales"]:
        L.append(
            f"| {s:g} | "
            + " | ".join(f"{a['scale'][c['key']][f'{s:g}']['psnr']:.2f}" for c in a["cells"])
            + " |"
        )
    L.append("")
    L.append(
        "Fitted exponent of the *excess* MSE over the noiseless value, "
        "`MSE(s) − MSE(0) ∝ s^k`: "
        + ", ".join(
            f"{c['model']} {c['n_hidden']} k={a['scale_exponent'][c['key']]:.2f}"
            for c in a["cells"]
        )
        + ".\n"
    )
    L.append("## A3. Shot noise\n")
    L.append(
        f"`ŝ(t) = Binomial(N, P(out=1))/N` per window, {a['shot_reps']} repetitions; the "
        "`predicted` column is the closed form `PSNR = 10 log10(R²/(MSE_∞ + E[p(1−p)]/N))`.\n"
    )
    budgets = [n for n in meta["argv"]["shots"] if n > 0]
    L.append(
        "| cell | source | "
        + " | ".join(f"N={n}" for n in budgets)
        + " | N→∞ | shots for ≤1 dB loss |"
    )
    L.append("|---" * (len(budgets) + 4) + "|")
    for c_ in a["cells"]:
        for source in ("clean", "noisy"):
            st = a["shots"][c_["key"]][source]
            got = {r["shots"]: r for r in st["rows"]}
            cols = " | ".join(_shot_cell(got.get(n)) for n in budgets)
            L.append(
                f"| {c_['model']} {c_['n_hidden']} | {source} | {cols} | "
                f"{st['exact_psnr']:.2f} | {st['shots_for_1db']:.0f} |"
            )
    L.append("")
    L.append("![A2](figA2_ecg_scale_shots.png)\n")

    if b:
        L.append(f"## B. Iris ({b['n_qubits']} qubits) under vendor noise\n")
        first = b["per_device"]["prototype"][b["devices"][0]]["reps"]
        be_name = first[0]["backend"] if first else "trajectories"
        L.append(f"Monte-Carlo trajectory backend `{be_name}`")
        L.append(
            f", **{b['trajectories']} trajectories** per evaluation "
            + (
                f"(requested {b['requested_trajectories']}, reduced to fit the "
                f"{meta['iris_budget_s']:.0f} s budget: one evaluation costs "
                f"{b['probe']['s_per_trajectory']:.3f} s per trajectory at "
                f"{b['n_qubits']} qubits and {b['probe']['n_evaluations_planned']} evaluations "
                "are needed)"
                if b["reduced"]
                else ""
            )
            + f", {b['reps']} independent trajectory seeds as the error bar.  "
            "**[ASSUMPTION] state preparation of the dataset superposition is perfect** — the "
            "20-code test superposition enters as `Circuit.initial_state` and only the network's "
            "own gates carry noise, so these numbers are optimistic.\n"
        )
        L.append("| device | proto acc | proto KL | classic acc | classic KL |")
        L.append("|---|---|---|---|---|")
        L.append(
            f"| *clean simulator* | {b['clean']['prototype']['accuracy']:.3f} | "
            f"{b['clean']['prototype']['kl']:.4f} | {b['clean']['classic']['accuracy']:.3f} | "
            f"{b['clean']['classic']['kl']:.4f} |"
        )
        for d in b["devices"]:
            p_, cl = b["per_device"]["prototype"][d], b["per_device"]["classic"][d]
            L.append(
                f"| {d} | {p_['accuracy_mean']:.3f}±{p_['accuracy_sd']:.3f} | "
                f"{p_['kl_mean']:.4f}±{p_['kl_sd']:.4f} | "
                f"{cl['accuracy_mean']:.3f}±{cl['accuracy_sd']:.3f} | "
                f"{cl['kl_mean']:.4f}±{cl['kl_sd']:.4f} |"
            )
        L.append("")
        L.append(f"`{a['scale_device']}` scaled:\n")
        L.append("| s | proto acc | proto KL | classic acc | classic KL |")
        L.append("|---|---|---|---|---|")
        for s in a["scales"]:
            key = f"{s:g}"
            if key not in b["scale"]["prototype"]:
                continue
            p_, cl = b["scale"]["prototype"][key], b["scale"]["classic"][key]
            L.append(
                f"| {s:g} | {p_['accuracy_mean']:.3f} | {p_['kl_mean']:.4f} | "
                f"{cl['accuracy_mean']:.3f} | {cl['kl_mean']:.4f} |"
            )
        L.append("")
        L.append("![B](figB_iris.png)\n")

    if c:
        L.append(f"## C. Noise-aware fine-tuning (ECG, 7 qubits, 1 prototype, `{c['device']}`)\n")
        L.append(
            f"{c['sweeps']} KW–SA coordinate sweeps ({c['evaluations']} noisy loss evaluations, "
            f"{c['subsample']} training windows each, {c['seconds']:.0f} s) starting from the "
            "trained parameters, loss on `NoisyBackend(numpy density matrix, "
            f"{c['device']})`.\n"
        )
        L.append("| parameters | PSNR under noise | PSNR noiseless | MSE under noise |")
        L.append("|---|---|---|---|")
        for name in ("before", "after"):
            r = c[name]
            L.append(
                f"| {name} | {r['noisy']['psnr']:.2f} | {r['clean']['psnr']:.2f} | "
                f"{r['noisy']['mse']:.3e} |"
            )
        L.append("")
        L.append(
            f"Recovery: **{c['after']['noisy']['psnr'] - c['before']['noisy']['psnr']:+.2f} dB** "
            "under noise, "
            f"**{c['after']['clean']['psnr'] - c['before']['clean']['psnr']:+.2f} dB** noiseless.\n"
        )
        L.append("![C](figC_finetune.png)\n")

    L.append("## Interpretation\n")
    L.extend(res["interpretation"])
    path.write_text("\n".join(L) + "\n")


def interpret(res: dict[str, Any], meta: dict[str, Any]) -> list[str]:
    a = res["A"]
    b = res.get("B")
    c = res.get("C")
    out: list[str] = []
    losses_db = {
        d: float(
            np.mean(
                [
                    a["clean"][k["key"]]["psnr"] - a["per_device"][k["key"]][d]["psnr"]
                    for k in a["cells"]
                ]
            )
        )
        for d in a["devices"]
    }
    best = min(losses_db, key=lambda d: losses_db[d])
    worst = max(losses_db, key=lambda d: losses_db[d])
    keep_all = [
        d
        for d in a["devices"]
        if all(a["per_device"][k["key"]][d]["psnr"] >= PSNR_TARGET_DB for k in a["cells"])
    ]
    out.append(
        f"* **ECG denoising is robust to vendor noise.**  Averaged over the six cells the PSNR "
        f"loss ranges from {losses_db[best]:.2f} dB (`{best}`) to {losses_db[worst]:.2f} dB "
        f"(`{worst}`).  The circuits are shallow — 22 to 54 gates, "
        f"{a['n_2q_gates'][0]}–{a['n_2q_gates'][-1]} two-qubit gates — which is why a task that "
        "needs 30 CRX in the bit-count study degrades much more there."
    )
    out.append(
        f"* **Devices that keep every cell above {PSNR_TARGET_DB:g} dB:** "
        + (", ".join(f"`{d}`" for d in keep_all) if keep_all else "none")
        + ".  Devices where at least the best cell clears it: "
        + ", ".join(
            f"`{d}`"
            for d in a["devices"]
            if any(a["per_device"][k["key"]][d]["psnr"] >= PSNR_TARGET_DB for k in a["cells"])
        )
        + f".  The un-denoised input scores {a['input_psnr']:.2f} dB, so a device below that "
        "value is actively harmful."
    )
    if b:
        acc_drop = {
            k: b["clean"][k]["accuracy"]
            - float(np.mean([b["per_device"][k][d]["accuracy_mean"] for d in b["devices"]]))
            for k in KINDS
        }
        best_dev = max(b["devices"], key=lambda d: b["per_device"]["prototype"][d]["accuracy_mean"])
        out.append(
            f"* **Iris does not survive 17-qubit depth under noise as well as ECG does.**  The "
            f"prototype network's clean test accuracy {b['clean']['prototype']['accuracy']:.2f} "
            f"falls by {acc_drop['prototype']:.2f} on average over the presets (classic: "
            f"{b['clean']['classic']['accuracy']:.2f} → −{acc_drop['classic']:.2f}); the best "
            f"preset is `{best_dev}` at "
            f"{b['per_device']['prototype'][best_dev]['accuracy_mean']:.2f}.  The circuit is "
            f"{b['n_qubits']} qubits deep with 48 (prototype) / 48 (classic) CRX, and — unlike "
            "the ECG read-out, which is a single-qubit activation — the Iris read-out is a "
            "*conditional* distribution over 4096 input strings, so depolarising noise that "
            "leaks probability out of the 20-code support hits it directly.  With perfect state "
            "preparation assumed, this is a lower bound on the damage."
        )
    shots_1db = {k["key"]: a["shots"][k["key"]]["clean"]["shots_for_1db"] for k in a["cells"]}
    positive = [r["shots"] for r in a["shots"][a["cells"][0]["key"]]["clean"]["rows"] if r["shots"]]
    small_n = min(positive) if positive else 0
    small_psnr = (
        min(
            r["psnr_mean"]
            for k in a["cells"]
            for r in a["shots"][k["key"]]["clean"]["rows"]
            if r["shots"] == small_n
        )
        if small_n
        else float("nan")
    )
    out.append(
        "* **Shot budget.**  Keeping the PSNR within 1 dB of the infinite-shot value needs "
        + ", ".join(
            f"{shots_1db[k['key']]:.0f} shots/window for {k['model']} {k['n_hidden']}"
            for k in a["cells"]
        )
        + ".  The requirement scales as `1/MSE_∞`, so the better cells are the expensive ones: "
        f"at {small_n} shots every cell collapses towards {small_psnr:.1f} dB, "
        "which is the shot-noise floor, not the model's."
    )
    if c:
        out.append(
            f"* **Noise-aware fine-tuning** of the 7-qubit single-prototype cell recovers "
            f"{c['after']['noisy']['psnr'] - c['before']['noisy']['psnr']:+.2f} dB under "
            f"`{c['device']}` in {c['sweeps']} sweeps, at a cost of "
            f"{c['after']['clean']['psnr'] - c['before']['clean']['psnr']:+.2f} dB on a "
            "noiseless simulator — the same trade the bit-count study saw: the channel is folded "
            "into the angles."
        )
    return out


# =================================================================================================
# driver
# =================================================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_out = "experiments/runs/ecg_iris_noise/" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--devices", type=str, default="all", help="'all' or a comma-separated list")
    p.add_argument("--scales", type=parse_floats, default=[0.0, 0.5, 1.0, 2.0, 4.0])
    p.add_argument("--shots", type=parse_ints, default=[0, 1000, 10000])
    p.add_argument("--trajectories", type=int, default=2000)
    p.add_argument("--out", type=str, default=default_out)
    p.add_argument("--studies", type=str, default="A,B,C")
    p.add_argument("--ecg-run", type=str, default="experiments/runs/ecg/20260820T162014")
    p.add_argument("--iris-run", type=str, default="experiments/runs/iris/20260820T144849Z")
    p.add_argument("--scale-device", type=str, default="ibm_fez")
    p.add_argument("--jobs", type=int, default=32)
    p.add_argument("--shot-reps", type=int, default=5)
    p.add_argument("--traj-reps", type=int, default=2)
    p.add_argument("--ecg-windows", type=int, default=0, help="0 = fit --ecg-budget-s")
    p.add_argument("--ecg-budget-s", type=float, default=1100.0)
    p.add_argument("--iris-budget-s", type=float, default=1500.0)
    p.add_argument("--iris-backend", choices=("gpu", "cpu"), default="gpu")
    p.add_argument("--iris-batch", type=int, default=64)
    p.add_argument("--iris-min-trajectories", type=int, default=64)
    p.add_argument("--iris-probe-trajectories", type=int, default=16)
    p.add_argument("--ft-device", type=str, default="ibm_fez")
    p.add_argument("--ft-sweeps", type=int, default=5)
    p.add_argument("--ft-subsample", type=int, default=96)
    p.add_argument("--ft-c0", type=float, default=0.2)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wall_t0 = time.perf_counter()
    studies = {s.upper() for s in parse_list(args.studies)}

    devices = presets.available_devices() if args.devices == "all" else parse_list(args.devices)
    unknown = [d for d in devices if d not in presets.available_devices()]
    if unknown:
        raise SystemExit(f"unknown device(s) {unknown}; have {presets.available_devices()}")
    if args.scale_device not in devices:
        # s = 1 of the scale sweep *is* the device row, and both studies read it from the same
        # cache; keeping the sweep device out of --devices would silently split the two.
        raise SystemExit(f"--scale-device {args.scale_device!r} must be among --devices")

    ecg_run = Path(args.ecg_run)
    ecg_meta = json.loads((ecg_run / "meta.json").read_text())
    ecg_res = json.loads((ecg_run / "results.json").read_text())
    npz = np.load(ecg_run / "params.npz")
    ds = ecg_train.prepare(
        source=ecg_meta["source"],
        n_samples=ecg_meta["n_samples"],
        seed=ecg_meta["seed"],
        sigma=ecg_meta["sigma"],
        train_frac=ecg_meta["train_frac"],
    )
    if not np.allclose(ds.clean, npz["clean"]):
        raise SystemExit("dataset does not reproduce the ECG run: check meta.json")
    encoding = ecg_meta["encoding"]
    cells = [
        {
            "key": f"{row['kind']}_{row['n_hidden']}",
            "kind": row["kind"],
            "model": row["model"],
            "n_hidden": row["n_hidden"],
            "n_qubits": row["n_qubits"],
            "params": npz[f"{row['kind']}_{row['n_hidden']}"].tolist(),
            "paper_block_psnr": row["psnr_test"],
        }
        for row in ecg_res["rows"]
    ]

    # --- ECG budget probe ------------------------------------------------------------------------
    scales_extra = [s for s in args.scales if s > 0 and s != 1.0]
    n_combos = len(devices) + len(scales_extra)
    units = cost_units([c["n_hidden"] for c in cells])
    probe_cell = max(cells, key=lambda c: c["n_hidden"])
    probe_net = _ecg_net(probe_cell)
    probe_params = np.asarray(probe_cell["params"], dtype=np.float64)
    _PROBE.update(
        net=probe_net,
        params=probe_params,
        X=ds.X[ds.test_idx[: args.jobs]],
        device=devices[0],
        encoding=encoding,
    )
    t0 = time.perf_counter()
    with mp.get_context("fork").Pool(processes=args.jobs) as pool:
        pool.map(_probe_job, range(args.jobs))
    probe_s = (time.perf_counter() - t0) / args.jobs
    n_test = int(ds.test_idx.size)
    full_estimate = n_test * n_combos * units * probe_s
    n_windows = (
        min(args.ecg_windows, n_test)
        if args.ecg_windows > 0
        else min(n_test, budget_windows(args.ecg_budget_s, probe_s, n_combos, units))
    )
    block = centred_block(ds.test_idx, n_windows)
    print(
        f"[budget] one {probe_net.n_qubits}-qubit noisy window costs {probe_s:.3f} s at "
        f"{args.jobs}-way parallelism; {n_combos} combos × {units:.2f} cell-units × "
        f"{n_test} test windows ≈ {full_estimate / 60:.0f} min → evaluating {n_windows} windows "
        f"(samples {block[0]}–{block[-1]})",
        flush=True,
    )

    X = ds.X[block]
    y = ds.y[block]
    R = ds.R

    # --- launch study B in a spawn process (GPU), concurrently with A ----------------------------
    iris_proc = None
    iris_json = out / "iris.json"
    if "B" in studies:
        iris_meta = json.loads((Path(args.iris_run) / "meta.json").read_text())
        job = {
            "seed": iris_meta["seed"],
            "train_size": float(iris_meta["split"]["train_size"]),
            "config": iris_meta["config"],
            "params": {k: np.load(Path(args.iris_run) / f"params-{k}.npy").tolist() for k in KINDS},
            "devices": devices,
            "scales_extra": scales_extra,
            "scale_device": args.scale_device,
            "trajectories": args.trajectories,
            "min_trajectories": args.iris_min_trajectories,
            "probe_trajectories": args.iris_probe_trajectories,
            "budget_s": args.iris_budget_s,
            "reps": args.traj_reps,
            "batch": args.iris_batch,
            "backend": args.iris_backend,
        }
        ctx = mp.get_context("spawn")
        iris_proc = ctx.Process(target=iris_worker, args=(job, str(iris_json)), daemon=False)
        iris_proc.start()
        print(f"[B] Iris study started in pid {iris_proc.pid} ({args.iris_backend})", flush=True)

    # --- worker state ----------------------------------------------------------------------------
    ft_rng = np.random.default_rng(0)
    ft_idx = np.sort(
        ft_rng.choice(
            ds.train_idx.size, size=min(args.ft_subsample, ds.train_idx.size), replace=False
        )
    )
    ft_windows = ds.train_idx[ft_idx]
    ft_cell = next(c for c in cells if c["kind"] == "prototype" and c["n_hidden"] == 1)
    _ECG.update(
        cells=cells,
        X=X,
        encoding=encoding,
        ft_net=_ecg_net(ft_cell),
        ft_X=ds.X[ft_windows],
        ft_y=ds.y[ft_windows],
        ft_device=args.ft_device,
    )

    a: dict[str, Any] = {
        "devices": devices,
        "scales": args.scales,
        "scale_device": args.scale_device,
        "cells": [{k: v for k, v in c.items() if k != "params"} for c in cells],
        "n_windows": int(n_windows),
        "n_test_windows": n_test,
        "n_samples": int(ds.clean.size),
        "block": [int(block[0]), int(block[-1])],
        "sigma": ds.sigma,
        "R": R,
        "backend": ref.DensityMatrixBackend.name,
        "probe_s": probe_s,
        "n_combos": n_combos,
        "full_estimate_s": full_estimate,
        "shot_reps": args.shot_reps,
        "input_psnr": ecg_model.psnr(y, ds.X[block][:, 2], R),
        "input_ssim_windowed": ecg_model.ssim_windowed(y, ds.X[block][:, 2]),
        "n_2q_gates": sorted(
            sum(
                1
                for g in _ecg_net(c).circuit(np.zeros(len(c["params"]))).gates()
                if g.n_qubits == 2
            )
            for c in cells
        ),
        "clean": {},
        "per_device": {},
        "scale": {},
        "scale_exponent": {},
        "shots": {},
    }
    clean_hat: dict[str, Real] = {}
    for c_ in cells:
        yh = ecg_model.predict(_ecg_net(c_), np.asarray(c_["params"]), X, None, encoding)
        clean_hat[c_["key"]] = yh
        a["clean"][c_["key"]] = ecg_metrics(y, yh, yh, R)
        a["per_device"][c_["key"]] = {}
        a["scale"][c_["key"]] = {"0": a["clean"][c_["key"]]}

    # --- A: the grid -----------------------------------------------------------------------------
    chunk = max(1, n_windows // 12)
    jobs: list[tuple[int, str, float, int, int]] = []
    for ci in range(len(cells)):
        for device in devices:
            jobs += [
                (ci, device, 1.0, i, min(i + chunk, n_windows)) for i in range(0, n_windows, chunk)
            ]
        for s in scales_extra:
            jobs += [
                (ci, args.scale_device, s, i, min(i + chunk, n_windows))
                for i in range(0, n_windows, chunk)
            ]
    jobs.sort(key=lambda j: -cells[j[0]]["n_hidden"])  # longest jobs first: better makespan
    acc: dict[tuple[int, str, float], Real] = {}
    t0 = time.perf_counter()
    done = 0
    with mp.get_context("fork").Pool(processes=args.jobs) as pool:
        for ci, device, s, i0, vals in pool.imap_unordered(_ecg_job, jobs, chunksize=1):
            arr = acc.setdefault((ci, device, s), np.full(n_windows, np.nan))
            arr[i0 : i0 + len(vals)] = vals
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(f"[A] {done}/{len(jobs)} chunks, {time.perf_counter() - t0:.0f}s", flush=True)
    a_seconds = time.perf_counter() - t0

    for (ci, device, s), yh in acc.items():
        key = cells[ci]["key"]
        m = ecg_metrics(y, yh, clean_hat[key], R)
        if s == 1.0 and device in devices:
            a["per_device"][key][device] = m
        if device == args.scale_device:
            a["scale"][key][f"{s:g}"] = m
    for c_ in cells:
        key = c_["key"]
        a["scale"][key]["1"] = a["per_device"][key][args.scale_device]
        a["scale_exponent"][key] = loglog_exponent(
            args.scales,
            [a["scale"][key][f"{s:g}"]["mse"] for s in args.scales],
            a["clean"][key]["mse"],
        )

    # --- A3: shot noise (free — resamples the activations already computed) ----------------------
    for c_ in cells:
        key = c_["key"]
        p_noisy = acc[(cells.index(c_), args.scale_device, 1.0)]
        a["shots"][key] = {
            "clean": shot_study(clean_hat[key], y, R, args.shots, args.shot_reps, 12345),
            "noisy": shot_study(p_noisy, y, R, args.shots, args.shot_reps, 54321),
        }

    results: dict[str, Any] = {"A": a}
    print(f"[A] done in {a_seconds:.0f}s", flush=True)

    # --- C: noise-aware fine-tuning --------------------------------------------------------------
    if "C" in studies:
        net_ft = _ECG["ft_net"]
        p0 = np.asarray(ft_cell["params"], dtype=np.float64)
        ft_chunk = max(1, len(ft_windows) // args.jobs)
        slices = [
            (i, min(i + ft_chunk, len(ft_windows))) for i in range(0, len(ft_windows), ft_chunk)
        ]
        t0 = time.perf_counter()
        with mp.get_context("fork").Pool(processes=args.jobs) as pool:

            def loss(params: Real) -> float:
                payloads = [([float(v) for v in params], i0, i1) for i0, i1 in slices]
                res = pool.map(_ft_chunk, payloads)
                return float(sum(r[0] for r in res) / sum(r[1] for r in res))

            cfg = kwsa.KWSAConfig(
                iterations=args.ft_sweeps,
                a0=ecg_train.DEFAULT_CONFIG.a0,
                alpha=ecg_train.DEFAULT_CONFIG.alpha,
                # [ASSUMPTION] c0 = 0.2 rather than the from-scratch 0.4: fine-tuning starts at a
                # trained point, so the finite-difference probe — which is also the move KW-SA
                # keeps when the gradient step overshoots — must be finer than the initial sweep.
                c0=args.ft_c0,
                gamma=ecg_train.DEFAULT_CONFIG.gamma,
                t0=0.0,
                seed=0,
            )
            res_ft = kwsa.minimise(loss, p0, cfg)
        ft_seconds = time.perf_counter() - t0
        noisy_be = noisy_backend(args.ft_device, 1.0)
        before_after: dict[str, Any] = {}
        for name, params in (("before", p0), ("after", res_ft.params)):
            yh_n = np.asarray(
                [v for v in _predict_pool(net_ft, params, X, args.ft_device, encoding, args.jobs)]
            )
            yh_c = ecg_model.predict(net_ft, params, X, None, encoding)
            before_after[name] = {
                "noisy": ecg_metrics(y, yh_n, clean_hat[ft_cell["key"]], R),
                "clean": ecg_metrics(y, yh_c, clean_hat[ft_cell["key"]], R),
                "params": [float(v) for v in params],
            }
        results["C"] = {
            "device": args.ft_device,
            "sweeps": args.ft_sweeps,
            "subsample": len(ft_windows),
            "evaluations": int(res_ft.evaluations),
            "seconds": ft_seconds,
            "history": [float(v) for v in res_ft.history],
            "backend": noisy_be.name,
            **before_after,
        }
        print(
            f"[C] fine-tuning: noisy PSNR "
            f"{before_after['before']['noisy']['psnr']:.2f} → "
            f"{before_after['after']['noisy']['psnr']:.2f} dB "
            f"({res_ft.evaluations} evals, {ft_seconds:.0f}s)",
            flush=True,
        )

    # --- join B ----------------------------------------------------------------------------------
    if iris_proc is not None:
        print("[B] waiting for the Iris process...", flush=True)
        iris_proc.join()
        if iris_json.exists():
            results["B"] = json.loads(iris_json.read_text())
            results["B"]["backend_kind"] = args.iris_backend
        else:
            print(f"[B] FAILED (exit {iris_proc.exitcode}); study B omitted", flush=True)

    results["interpretation"] = interpret(results, {"jobs": args.jobs})
    meta = {
        "experiment": "ecg_iris_noise",
        "out": str(out),
        "git_hash": git_hash(),
        "timestamp": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "argv": vars(args) | {"out": str(out)},
        "ecg_run": str(ecg_run),
        "iris_run": str(args.iris_run),
        "ecg_run_git": ecg_meta["git_hash"],
        "seeds": {
            "ecg_dataset": ecg_meta["seed"],
            "shot_clean": 12345,
            "shot_noisy": 54321,
            "trajectory": [1000 + r * 7919 for r in range(args.traj_reps)],
            "finetune": 0,
        },
        "backends": {
            "ecg": ref.DensityMatrixBackend.name,
            "iris": args.iris_backend,
        },
        "jobs": args.jobs,
        "iris_budget_s": args.iris_budget_s,
        "timings": {
            "probe_s": probe_s,
            "A_s": a_seconds,
            "C_s": results.get("C", {}).get("seconds", 0.0),
            "B_s": results.get("B", {}).get("wall_seconds", 0.0),
        },
        "wall_seconds": time.perf_counter() - wall_t0,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    fig_devices(results, out / "figA_ecg_devices.png")
    fig_scale_and_shots(results, out / "figA2_ecg_scale_shots.png")
    fig_iris(results, out / "figB_iris.png")
    fig_finetune(results, out / "figC_finetune.png")
    write_report(results, meta, out / "results.md")
    print(f"\nwall {meta['wall_seconds']:.0f}s → {out}")
    return 0


def _predict_pool(
    net: Network, params: Real, X: Real, device: str, encoding: str, jobs: int
) -> Real:
    """``predict`` on the noisy backend, spread over a fresh fork pool (study C's scoring)."""
    prev = dict(_ECG)
    _ECG.update(ft_net=net, ft_X=X, ft_y=np.zeros(len(X)), ft_device=device, encoding=encoding)
    chunk = max(1, len(X) // jobs)
    slices = [(i, min(i + chunk, len(X))) for i in range(0, len(X), chunk)]
    try:
        with mp.get_context("fork").Pool(processes=jobs) as pool:
            parts = pool.map(
                _predict_chunk, [([float(v) for v in params], i0, i1) for i0, i1 in slices]
            )
    finally:
        _ECG.clear()
        _ECG.update(prev)
    return np.concatenate(parts)


def _predict_chunk(payload: tuple[list[float], int, int]) -> Real:
    params, i0, i1 = payload
    return ecg_model.predict(
        _ECG["ft_net"],
        np.asarray(params, dtype=np.float64),
        _ECG["ft_X"][i0:i1],
        noisy_backend(_ECG["ft_device"], 1.0),
        _ECG["encoding"],
    )


if __name__ == "__main__":
    raise SystemExit(main())
