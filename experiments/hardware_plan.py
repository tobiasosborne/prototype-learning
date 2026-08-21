"""Every number `reports/hardware-plan.md` quotes, regenerated from code and run data.

Review 002 blocker 1: the hardware plan cited no run dir and a dozen of its numbers were wrong
(TVD per device, P(out=1), "lowest 2q error", the dense state-prep CNOT count, SWAP fractions,
the Garnet T2 sentence, a $ figure).  This script is the single source for the report's
quantitative claims: it writes ``experiments/runs/hardware-plan/<stamp>/{numbers.json,
numbers.md, meta.json}`` and the report quotes *only* what ``numbers.md`` prints.

Sections (the keys of ``numbers.json``):

* ``logical``   — qubits / depth / CRX / 1q of each paper network (`Network.circuit`).
* ``compiled``  — `transpile.compile_for` against each *real* preset
  (`ptqnn/noise/devices/*.toml`): n_2q, n_swap, depth, critical path (+ readout), SWAP share.
* ``fidelity``  — the multiplicative EPLG-style estimate
  ``F ≈ Π(1−ε_2q)·Π(1−ε_1q)·Π_q F_idle(t_crit)·Π_q(1−ε_ro)`` per device for Flappy 7 q and
  bit-count 11 q, with every factor printed, plus the *simulated* total-variation distance
  between the compiled circuit run clean and run under ``DeviceSpec.noise_model()``
  (density matrix, exact).  Idle error is the average-gate infidelity of thermal relaxation,
  ``F_avg = ½ + (2e^{−t/T2} + e^{−t/T1})/6``; where no T1/T2 is published (Quantinuum) the
  *Memory Error per Depth-1 Circuit Time* ``m`` from the TOML provenance is charged as
  ``(1−m)^(n_qubits·depth)`` ``[ASSUMPTION]``.
* ``sabre``     — qiskit's own transpiler (`IBMRuntimeExecutor`) at optimisation levels 1–3 on
  the same heavy-hex patch, with the critical path of the ISA circuit walked with ibm_fez's
  durations, and the same fidelity estimate.
* ``cost``      — `executor.cost_estimate` per device at the report's shot counts, plus the
  Braket / Azure formulas the report prints.
* ``shots``     — ``P(out=1)`` of the trained bit-count model (params from the bit-count noise
  run), the shot-noise KL floor ``(K−1)/(2·N·P(out=1))``, and the Flappy decision-margin
  quantiles and predicted flip rates copied from the Flappy noise run.
* ``noise_run`` — TVD / KL per device for the trained bit-count model, copied from
  ``experiments/runs/bitcount_noise/<stamp>/results.json`` (not recomputed).
* ``stateprep`` — dense / sparse CNOT counts for the Iris dataset state.

Fail fast, log eagerly: every device/circuit pair prints a line with its wall time the moment
it finishes (``flush=True``), so a blow-out is visible within seconds.

    LD_LIBRARY_PATH=/usr/lib/wsl/lib uv run python experiments/hardware_plan.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ptqnn.backends import reference as ref
from ptqnn.circuits import ir
from ptqnn.circuits.neurons import Network
from ptqnn.hardware import executor as ex
from ptqnn.hardware import transpile as tp
from ptqnn.noise import presets
from ptqnn.noise.model import NoisyBackend
from ptqnn.tasks import bitcount
from ptqnn.tasks.flappy import agent as flappy_agent

ROOT = Path(__file__).resolve().parents[1]
BITCOUNT_NOISE_RUN = ROOT / "experiments/runs/bitcount_noise/20260820T145335Z"
FLAPPY_NOISE_RUN = ROOT / "experiments/runs/flappy_noise/20260820T154249Z"
FLAPPY_TRAIN_RUN = ROOT / "experiments/runs/flappy/20260820T143753Z"

# Paper network sizes: Flappy Network(2,4) = 7 q [PTQNN Fig 10]; ECG 5 inputs + 1–3 prototypes
# [PTQNN Fig 12]; bit-count Network(5,5) = 11 q [PTQNN Fig 5] / Network(5,6) = 12 q [Fig 4];
# Iris 12 input qubits [PTQNN §III.C] with 4 (our default) or 12 hidden.
NETWORKS: dict[str, tuple[int, int]] = {
    "flappy_7q": (2, 4),
    "ecg_7q": (5, 1),
    "ecg_8q": (5, 2),
    "ecg_9q": (5, 3),
    "bitcount_11q": (5, 5),
    "bitcount_12q": (5, 6),
    "iris_17q": (12, 4),
    "iris_25q": (12, 12),
}
FIDELITY_CIRCUITS = ("flappy_7q", "bitcount_11q")
SHOTS_FOR_COST = {"flappy_7q": 4_000, "bitcount_11q": 64_000}
# [ASSUMPTION] IQM publishes no 1q gate duration and Rigetti publishes none for 1q; the TOMLs
# carry 0 ns, which would make the critical path (and hence the idle error) an under-count.  We
# substitute 40 ns for both — IQM's Radiance data sheet bounds PRX at 40 ns, Rigetti's ±π/2 RX is
# of the same order — and record the substitution in ``meta.json`` and in every affected row.
ASSUMED_1Q_NS: dict[str, float] = {"iqm_garnet": 40.0, "rigetti_ankaa_3": 40.0}
# Flappy observations used for the simulated TVD: grid corners and centre (bird height, gap).
FLAPPY_OBS = ((0.5, 0.5), (0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def git_hash() -> str:
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


# --- inputs -------------------------------------------------------------------------------------


def trained_bitcount_params() -> tuple[np.ndarray, str]:
    r = json.loads((BITCOUNT_NOISE_RUN / "results.json").read_text())
    p = np.asarray(r["A"]["prototype"]["params"], dtype=np.float64)
    if p.shape != (r["n_params"],):
        raise RuntimeError(f"bit-count params shape {p.shape} != ({r['n_params']},)")
    return p, f"{BITCOUNT_NOISE_RUN.relative_to(ROOT)}/results.json A/prototype/params"


def trained_flappy_params() -> tuple[np.ndarray, dict[str, Any], str]:
    p = np.load(FLAPPY_TRAIN_RUN / "params.npy").astype(np.float64)
    meta = json.loads((FLAPPY_TRAIN_RUN / "meta.json").read_text())
    if p.shape != (meta["network"]["n_params"],):
        raise RuntimeError(f"flappy params shape {p.shape}")
    return p, meta, f"{FLAPPY_TRAIN_RUN.relative_to(ROOT)}/params.npy"


def circuit_for(
    name: str, params_by_name: dict[str, np.ndarray], rng: np.random.Generator
) -> ir.Circuit:
    n_in, n_hid = NETWORKS[name]
    net = Network(n_inputs=n_in, n_hidden=n_hid, hidden="prototype")
    params = params_by_name.get(name)
    if params is None:
        params = net.random_params(rng)
    if name.startswith("bitcount"):
        prep = bitcount.superposition_prep(net)
    elif name.startswith("flappy"):
        prep = flappy_agent.encode(np.asarray(FLAPPY_OBS[0]), "amplitude", 1.0)
    else:
        # ECG / Iris: RX input rotations at a representative angle (the count is what matters).
        angles = rng.uniform(0.3, 2.8, n_in)
        prep = [ir.rx(q, float(a)) for q, a in zip(net.input_qubits, angles, strict=True)]
    return net.circuit(params, prep)


def memory_error(dev: presets.DeviceSpec) -> float | None:
    """Quantinuum's 'Memory Error per Depth-1 Circuit Time' from the TOML provenance block."""
    for p in dev.provenance:
        if "memory error" in str(p.get("field", "")).lower():
            return float(p["value"])
    return None


# --- physics helpers ----------------------------------------------------------------------------


def idle_fidelity(t_ns: float, t1_us: float, t2_us: float) -> float:
    """Average gate fidelity of thermal relaxation for time ``t``: ``½ + (2Λ + λ)/6`` with
    ``λ = e^{−t/T1}``, ``Λ = e^{−t/T2}`` (Nielsen's F_avg = (d·F_ent + 1)/(d+1), d = 2)."""
    lam = math.exp(-t_ns * 1e-3 / t1_us)
    big = math.exp(-t_ns * 1e-3 / t2_us)
    return 0.5 + (2.0 * big + lam) / 6.0


def fidelity_budget(
    dev: presets.DeviceSpec, n_1q: int, n_2q: int, depth: int, t_gate_ns: float | None, nq: int
) -> dict[str, Any]:
    f2 = (1.0 - dev.gate_2q_error) ** n_2q
    f1 = (1.0 - dev.gate_1q_error) ** n_1q if "gate_1q" not in dev.missing_fields else None
    mem = memory_error(dev)
    idle: float | None
    idle_how: str
    if dev.t1_us is not None and dev.t2_us is not None and t_gate_ns:
        idle = idle_fidelity(t_gate_ns, dev.t1_us, dev.t2_us) ** nq
        idle_how = f"F_avg(t={t_gate_ns:.4g} ns, T1={dev.t1_us:g} µs, T2={dev.t2_us:g} µs)^{nq}"
    elif mem is not None:
        idle = (1.0 - mem) ** (nq * depth)
        idle_how = f"(1 − {mem:g})^({nq}·{depth}) [ASSUMPTION: memory error per depth-1 time]"
    else:
        idle, idle_how = None, "no T1/T2 and no memory error published"
    if "readout" in dev.missing_fields:
        ro: float | None = None
    else:
        ro = (1.0 - 0.5 * (dev.readout_p01 + dev.readout_p10)) ** nq
    known = [f for f in (f2, f1, idle, ro) if f is not None]
    total = float(np.prod(known))
    return {
        "F_2q": f2,
        "F_1q": f1,
        "F_idle": idle,
        "F_idle_how": idle_how,
        "F_readout": ro,
        "F_total": total,
        "upper_bound_only": len(known) < 4,
    }


def pick_dm_backend() -> tuple[Any, str]:
    try:
        from ptqnn.backends.gpu import GpuDensityMatrixBackend

        be = GpuDensityMatrixBackend()
        be.probabilities(ir.Circuit(2, ops=[ir.h(0)]))
        return be, "cupy-density-matrix"
    except Exception as e:
        log(f"GPU density matrix unavailable ({type(e).__name__}: {e}); trying orkan")
    try:
        from ptqnn.backends.orkan import OrkanDensityMatrixBackend

        be = OrkanDensityMatrixBackend()
        be.probabilities(ir.Circuit(2, ops=[ir.h(0)]))
        return be, "orkan-density-matrix"
    except Exception as e:
        log(f"orkan unavailable ({type(e).__name__}: {e}); using numpy (slow)")
    return ref.DensityMatrixBackend(), "numpy-density-matrix"


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


# --- sections -----------------------------------------------------------------------------------


def section_logical(circuits: dict[str, ir.Circuit]) -> dict[str, Any]:
    out = {}
    for name, c in circuits.items():
        m = tp.metrics(c)
        out[name] = {
            "n_qubits": c.n_qubits,
            "depth": int(m["depth"] or 0),
            "n_crx": c.count("crx"),
            "n_1q": int(m["n_1q"] or 0),
        }
    return out


def section_compiled(
    circuits: dict[str, ir.Circuit], devices: dict[str, presets.DeviceSpec]
) -> tuple[dict[str, Any], dict[tuple[str, str], tp.CompiledCircuit]]:
    out: dict[str, Any] = {}
    compiled: dict[tuple[str, str], tp.CompiledCircuit] = {}
    for cname, c in circuits.items():
        out[cname] = {}
        for dname, dev in devices.items():
            if c.n_qubits > dev.n_qubits:
                out[cname][dname] = {"skipped": f"{c.n_qubits} q > {dev.n_qubits} q"}
                continue
            t0 = time.perf_counter()
            cc = tp.compile_for(c, dev)
            compiled[(cname, dname)] = cc
            m = cc.metrics
            n2 = int(m["n_2q"] or 0)
            ns = int(m["n_swap"] or 0)
            out[cname][dname] = {
                "n_1q": int(m["n_1q"] or 0),
                "n_2q": n2,
                "n_swap": ns,
                "swap_share_of_2q": (3 * ns / n2) if n2 else 0.0,
                "depth": int(m["depth"] or 0),
                "gate_time_ns": m.get("gate_time_ns"),
                "critical_path_ns": m.get("critical_path_ns"),
                "readout_ns": m.get("readout_ns"),
                "notes": list(cc.notes),
            }
            log(
                f"compile {cname} @ {dname}: n_2q={n2} swaps={ns} depth={m['depth']} "
                f"({time.perf_counter() - t0:.2f} s)"
            )
    return out, compiled


def section_fidelity(
    circuits: dict[str, ir.Circuit],
    devices: dict[str, presets.DeviceSpec],
    compiled: dict[tuple[str, str], tp.CompiledCircuit],
    params: dict[str, np.ndarray],
    dm_backend: Any,
    simulate: bool,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    sv = ref.StatevectorBackend()
    for cname in FIDELITY_CIRCUITS:
        out[cname] = {}
        for dname, dev in devices.items():
            cc = compiled.get((cname, dname))
            if cc is None:
                continue
            m = cc.metrics
            row = fidelity_budget(
                dev,
                int(m["n_1q"] or 0),
                int(m["n_2q"] or 0),
                int(m["depth"] or 0),
                m.get("gate_time_ns"),
                circuits[cname].n_qubits,
            )
            row["sim_tvd"] = None
            if simulate:
                t0 = time.perf_counter()
                model = dev.noise_model()
                noisy = NoisyBackend(dm_backend, model)
                if cname == "flappy_7q":
                    n_in, n_hid = NETWORKS[cname]
                    net = Network(n_inputs=n_in, n_hidden=n_hid, hidden="prototype")
                    tv = []
                    for obs in FLAPPY_OBS:
                        c = net.circuit(
                            params[cname], flappy_agent.encode(np.asarray(obs), "amplitude", 1.0)
                        )
                        cc_obs = tp.compile_for(c, dev)
                        tv.append(
                            tvd(
                                sv.probabilities(cc_obs.circuit),
                                noisy.probabilities(cc_obs.circuit),
                            )
                        )
                    row["sim_tvd"] = float(np.mean(tv))
                    row["sim_tvd_per_obs"] = [float(v) for v in tv]
                else:
                    row["sim_tvd"] = tvd(
                        sv.probabilities(cc.circuit), noisy.probabilities(cc.circuit)
                    )
                row["sim_seconds"] = time.perf_counter() - t0
                log(
                    f"fidelity {cname} @ {dname}: F={row['F_total']:.3f} "
                    f"sim TVD={row['sim_tvd']:.4f} ({row['sim_seconds']:.1f} s)"
                )
            else:
                log(f"fidelity {cname} @ {dname}: F={row['F_total']:.3f} (simulation skipped)")
            out[cname][dname] = row
    return out


def _qiskit_critical_path_ns(isa: Any, dev: presets.DeviceSpec) -> float:
    """ASAP critical path of a qiskit ISA circuit with the preset's durations (rz is virtual)."""
    finish = [0.0] * isa.num_qubits
    for inst in isa.data:
        qs = [isa.find_bit(q).index for q in inst.qubits]
        if inst.operation.name in ("rz", "barrier", "measure", "id"):
            dt = 0.0
        elif len(qs) == 1:
            dt = dev.gate_1q_ns
        else:
            dt = dev.gate_2q_ns
        start = max(finish[q] for q in qs)
        for q in qs:
            finish[q] = start + dt
    return max(finish)


def section_sabre(circuits: dict[str, ir.Circuit], fez: presets.DeviceSpec) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cname in FIDELITY_CIRCUITS:
        out[cname] = {}
        for level in (1, 2, 3):
            t0 = time.perf_counter()
            isa, _ = ex.IBMRuntimeExecutor(optimization_level=level, seed=0)._isa(circuits[cname])
            ops = dict(isa.count_ops())
            n2 = sum(v for k, v in ops.items() if k in ("cz", "ecr", "cx"))
            n1 = sum(v for k, v in ops.items() if k in ("sx", "x"))  # rz is virtual on Heron
            t_gate = _qiskit_critical_path_ns(isa, fez)
            nq = circuits[cname].n_qubits
            fb = fidelity_budget(fez, n1, n2, isa.depth(), t_gate, nq)
            out[cname][f"O{level}"] = {
                "n_2q": n2,
                "n_1q_physical": n1,
                "n_rz_virtual": int(ops.get("rz", 0)),
                "depth": isa.depth(),
                "gate_time_ns": t_gate,
                "critical_path_ns": t_gate + fez.readout_ns,
                **fb,
            }
            log(
                f"qiskit O{level} {cname}: n_2q={n2} depth={isa.depth()} t={t_gate:.0f} ns "
                f"F={fb['F_total']:.3f} ({time.perf_counter() - t0:.1f} s)"
            )
    return out


def section_cost(
    devices: dict[str, presets.DeviceSpec], compiled: dict[tuple[str, str], tp.CompiledCircuit]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cname, shots in SHOTS_FOR_COST.items():
        out[cname] = {"shots": shots}
        for dname, dev in devices.items():
            cc = compiled.get((cname, dname))
            if cc is None:
                continue
            est = ex.cost_estimate(cc, shots, dev)
            row = est.as_dict()
            m = cc.metrics
            n1, n2, nq = int(m["n_1q"] or 0), int(m["n_2q"] or 0), int(m["n_qubits"] or 0)
            # The formulas the report prints alongside the library estimate.
            if dname.startswith("quantinuum"):
                hqc = est.value
                row["usd_at_12_5_per_hqc"] = 12.5 * hqc
                row["hqc_per_shot"] = (n1 + 10 * n2 + 5 * 2 * nq) / 5000.0
            if dname == "ionq_forte":
                row["braket_usd"] = ex._BRAKET_PER_TASK + 0.08 * shots
            if dname == "ibm_fez":
                ns = float(m["critical_path_ns"] or 0.0)
                row["qpu_seconds_no_overhead"] = shots * ns * 1e-9
                row["usd_no_overhead_at_1_60_per_s"] = 1.60 * shots * ns * 1e-9
            out[cname][dname] = row
    return out


def section_shots(params: dict[str, np.ndarray]) -> dict[str, Any]:
    n_in, n_hid = NETWORKS["bitcount_11q"]
    net = Network(n_inputs=n_in, n_hidden=n_hid, hidden="prototype")
    sv = ref.StatevectorBackend()
    p = sv.probabilities(net.circuit(params["bitcount_11q"], bitcount.superposition_prep(net)))
    p_out1 = float(ref.marginal(p, [net.output_qubit])[1])
    k = 2**n_in
    floor_coeff = (k - 1) / (2.0 * p_out1)  # E[KL] ≈ coeff / N
    fl = json.loads((FLAPPY_NOISE_RUN / "results.json").read_text())
    m = fl["C"]["margins"]
    quant = dict(zip([str(q) for q in m["quantile_levels"]], m["quantiles"], strict=True))
    return {
        "bitcount_p_out1": p_out1,
        "bitcount_kl_floor_coefficient": floor_coeff,
        "bitcount_kl_floor_shots": {
            str(target): floor_coeff / target for target in (5e-3, 1e-3, 5.5e-4)
        },
        "flappy_margin_quantiles": quant,
        "flappy_margin_frames": m["frames"],
        "flappy_predicted_flip_rate": m["predicted_flip_rate"],
        "flappy_source": f"{FLAPPY_NOISE_RUN.relative_to(ROOT)}/results.json C/margins",
    }


def section_noise_run() -> dict[str, Any]:
    r = json.loads((BITCOUNT_NOISE_RUN / "results.json").read_text())
    out: dict[str, Any] = {"source": f"{BITCOUNT_NOISE_RUN.relative_to(ROOT)}/results.json A"}
    for kind in ("prototype", "classic"):
        a = r["A"][kind]
        out[kind] = {"clean_kl": a["clean_kl"], "per_device": {}}
        for d, row in a["per_device"].items():
            out[kind]["per_device"][d] = {
                k: row[k]
                for k in ("kl_target", "tv_target", "kl_clean_vs_noisy", "tv_clean_vs_noisy")
            }
    return out


def section_stateprep(n: int = 12, k_train: int = 120) -> dict[str, Any]:
    """Dense: Shende–Bullock–Markov ``2^{n+1} − 2n − 2`` CNOTs (Möttönen et al. is ``2^{n+1} − 2n``
    for ``n`` qubits); Plesch–Brukner ``≈ 23/24·2^n``.  Sparse (``k`` non-zero amplitudes):
    Gleinig–Hoefler ``O(k·n)``; we quote the leading term ``k·n`` with ``k = 120`` = Iris
    train split (``[ASSUMPTION]``; the constant is implementation-dependent)."""
    dense_sbm = 2 ** (n + 1) - 2 * n - 2
    dense_pb = 23 / 24 * 2**n
    sparse = k_train * n
    h2 = presets.device("quantinuum_h2_1")
    return {
        "n": n,
        "dense_cnots_sbm": dense_sbm,
        "dense_cnots_plesch_brukner": round(dense_pb),
        "sparse_cnots_leading_term": sparse,
        "k_train": k_train,
        "h2_1_fidelity_dense_sbm": (1 - h2.gate_2q_error) ** dense_sbm,
        "h2_1_fidelity_dense_pb": (1 - h2.gate_2q_error) ** dense_pb,
        "h2_1_fidelity_sparse": (1 - h2.gate_2q_error) ** sparse,
    }


# --- rendering ----------------------------------------------------------------------------------


def _f(x: Any, fmt: str = ".4g") -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return format(x, fmt)
    return str(x)


def render(num: dict[str, Any], devices: list[str]) -> str:
    L: list[str] = []
    w = L.append
    w("# Hardware-plan numbers\n")
    w("Generated by `experiments/hardware_plan.py`; do not edit by hand.\n")

    w("## logical\n")
    w("| circuit | qubits | depth | CRX | 1q |")
    w("|---|---:|---:|---:|---:|")
    for k, v in num["logical"].items():
        w(f"| {k} | {v['n_qubits']} | {v['depth']} | {v['n_crx']} | {v['n_1q']} |")

    w("\n## compiled (our greedy router; n_2q includes 3 per SWAP)\n")
    w("| circuit | " + " | ".join(devices) + " |")
    w("|---|" + "---|" * len(devices))
    for c, row in num["compiled"].items():
        cells = []
        for d in devices:
            r = row[d]
            if "skipped" in r:
                cells.append("n/a")
                continue
            t = r["critical_path_ns"]
            ts = "—" if t is None else (f"{t / 1e6:.1f} ms" if t > 1e6 else f"{t / 1e3:.1f} µs")
            cells.append(
                f"{r['n_2q']} ({r['n_swap']} sw, {100 * r['swap_share_of_2q']:.0f}%) "
                f"· d{r['depth']} · {ts}"
            )
        w(f"| {c} | " + " | ".join(cells) + " |")

    w(
        "\n## fidelity (multiplicative estimate; "
        "sim TVD = compiled circuit clean vs preset noise model)\n"
    )
    w(
        "Quantinuum presets carry no T1/T2, so their simulated TVD has *no* idle/memory error "
        "in it; the F_idle column charges it separately.  IQM/Rigetti 1q durations are the "
        "40 ns [ASSUMPTION].\n"
    )
    for c, rows in num["fidelity"].items():
        w(f"\n### {c}\n")
        w("| device | F_2q | F_1q | F_idle | idle model | F_ro | **F** | sim TVD |")
        w("|---|---:|---:|---:|---|---:|---:|---:|")
        for d, r in rows.items():
            tot = f"**{r['F_total']:.3f}**" + (" (upper bound)" if r["upper_bound_only"] else "")
            w(
                f"| {d} | {_f(r['F_2q'])} | {_f(r['F_1q'])} | {_f(r['F_idle'])} | "
                f"{r['F_idle_how']} | {_f(r['F_readout'])} | {tot} | {_f(r['sim_tvd'])} |"
            )

    w("\n## sabre (qiskit transpile on the heavy-hex patch, ibm_fez durations)\n")
    w("| circuit | level | n_2q | n_1q phys | depth | t_gate | F_2q | F_1q | F_idle | **F** |")
    w("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c, rows in num["sabre"].items():
        for lv, r in rows.items():
            w(
                f"| {c} | {lv} | {r['n_2q']} | {r['n_1q_physical']} | {r['depth']} | "
                f"{r['gate_time_ns'] / 1e3:.2f} µs | {r['F_2q']:.4f} | {r['F_1q']:.4f} | "
                f"{r['F_idle']:.4f} | **{r['F_total']:.3f}** |"
            )

    w("\n## cost\n")
    for c, rows in num["cost"].items():
        w(f"\n### {c} at {rows['shots']} shots\n")
        w("| device | model | value | unit | extra |")
        w("|---|---|---:|---|---|")
        for d, r in rows.items():
            if d == "shots":
                continue
            extra = {
                k: v
                for k, v in r.items()
                if k not in ("model", "unit", "value", "breakdown", "sources", "assumptions")
            }
            w(
                f"| {d} | {r['model']} | {r['value']:.4g} | {r['unit']} | "
                f"{', '.join(f'{k}={_f(v)}' for k, v in extra.items())} |"
            )

    s = num["shots"]
    w("\n## shots\n")
    w(f"- bit-count trained model P(out=1) = {s['bitcount_p_out1']:.4f}")
    w(
        f"- KL floor E[KL] ≈ {s['bitcount_kl_floor_coefficient']:.2f} / N; shots for floor: "
        + ", ".join(f"{k} → {v:,.0f}" for k, v in s["bitcount_kl_floor_shots"].items())
    )
    w(
        f"- Flappy margin quantiles over {s['flappy_margin_frames']:,} frames: "
        + ", ".join(
            f"p{int(float(q) * 100)}={v:.4f}" for q, v in s["flappy_margin_quantiles"].items()
        )
    )
    w(
        "- Flappy predicted flip rate: "
        + ", ".join(f"N={k}: {v:.3f}" for k, v in s["flappy_predicted_flip_rate"].items())
    )

    nr = num["noise_run"]
    w(f"\n## noise_run ({nr['source']})\n")
    w(
        "| device | proto TVD vs clean | proto TVD vs target | proto KL vs target | "
        "classic TVD vs clean | classic KL vs target |"
    )
    w("|---|---:|---:|---:|---:|---:|")
    for d in devices:
        p = nr["prototype"]["per_device"][d]
        c = nr["classic"]["per_device"][d]
        w(
            f"| {d} | {p['tv_clean_vs_noisy']:.4f} | {p['tv_target']:.4f} | "
            f"{p['kl_target']:.4f} | {c['tv_clean_vs_noisy']:.4f} | {c['kl_target']:.4f} |"
        )
    w(
        f"\nclean KL: prototype {nr['prototype']['clean_kl']:.3g}, "
        f"classic {nr['classic']['clean_kl']:.3g}"
    )

    sp = num["stateprep"]
    w("\n## stateprep (Iris, n = 12)\n")
    for k, v in sp.items():
        w(f"- {k}: {_f(v)}")

    w("\n## lowest published 2q error\n")
    for d, e in sorted(num["device_2q_error"].items(), key=lambda kv: kv[1]):
        w(f"- {d}: {e:g}")
    return "\n".join(L) + "\n"


# --- main ---------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--no-sim", action="store_true", help="skip the density-matrix TVD simulation")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/runs/hardware-plan")
    args = ap.parse_args(argv)

    t_start = time.perf_counter()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out / stamp
    out.mkdir(parents=True, exist_ok=False)
    log(f"run dir {out}")

    rng = np.random.default_rng(args.seed)
    bc_params, bc_src = trained_bitcount_params()
    fl_params, _fl_meta, fl_src = trained_flappy_params()
    params = {"bitcount_11q": bc_params, "flappy_7q": fl_params}
    circuits = {name: circuit_for(name, params, rng) for name in NETWORKS}

    names = presets.available_devices()
    devices = {n: presets.device(n, allow_missing=True) for n in names}
    for n, d in devices.items():
        if d.missing_fields:
            log(f"{n}: unpublished fields {d.missing_fields} — fidelity is an upper bound")
        if n in ASSUMED_1Q_NS:
            if d.gate_1q_ns:
                raise RuntimeError(
                    f"{n} now publishes a 1q duration ({d.gate_1q_ns} ns); drop ASSUMED_1Q_NS"
                )
            devices[n] = dataclasses.replace(d, gate_1q_ns=ASSUMED_1Q_NS[n])
            log(
                f"{n}: [ASSUMPTION] 1q duration {ASSUMED_1Q_NS[n]} ns substituted "
                "for the unpublished value"
            )

    dm_backend, dm_name = (None, "none") if args.no_sim else pick_dm_backend()
    log(f"density-matrix backend: {dm_name}")

    num: dict[str, Any] = {}
    num["logical"] = section_logical(circuits)
    num["compiled"], compiled = section_compiled(circuits, devices)
    num["fidelity"] = section_fidelity(
        circuits, devices, compiled, params, dm_backend, not args.no_sim
    )
    num["sabre"] = section_sabre(circuits, devices["ibm_fez"])
    num["cost"] = section_cost(devices, compiled)
    num["shots"] = section_shots(params)
    num["noise_run"] = section_noise_run()
    num["stateprep"] = section_stateprep()
    num["device_2q_error"] = {n: d.gate_2q_error for n, d in devices.items()}

    (out / "numbers.json").write_text(json.dumps(num, indent=1, default=float))
    (out / "numbers.md").write_text(render(num, names))
    meta = {
        "experiment": "hardware_plan",
        "timestamp_utc": stamp,
        "git_hash": git_hash(),
        "seed": args.seed,
        "backend": dm_name,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "assumed_1q_ns": ASSUMED_1Q_NS,
        "inputs": {
            "bitcount_params": bc_src,
            "flappy_params": fl_src,
            "flappy_noise_run": str(FLAPPY_NOISE_RUN.relative_to(ROOT)),
            "bitcount_noise_run": str(BITCOUNT_NOISE_RUN.relative_to(ROOT)),
        },
        "wall_seconds": time.perf_counter() - t_start,
        "argv": sys.argv[1:],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    log(f"done in {meta['wall_seconds']:.1f} s → {out / 'numbers.md'}")
    return out


if __name__ == "__main__":
    main()
