"""Kiefer–Wolfowitz finite-difference steps inside a simulated-annealing acceptance loop.

``[PTQNN §II.C]``: "The method estimates the gradient of an objective function from noisy
function evaluations ... ``x_{n+1} = x_n + a_n (M(x_n + c_n) − M(x_n − c_n)) / (2 c_n)`` ...
To further improve the exploration of the optimization landscape, we employ a simulated
annealing (SA) scheme ... ``P(e(x_t), e(x̂_t), T) ∝ exp(−(e(x̂_t) − e(x_t)) / kT)`` [Eq 3] ...
Candidate solutions that reduce the loss are therefore always accepted. Solutions with an
increased loss may also be accepted ... our implementation is solely based on stochastic
optimization."  ``[SNN §III.A]``: "the simulated annealing heuristic samples a neighboring state
x̂_t of the current state x_t, e.g. based on an approximate gradient."

Neither paper publishes ``a_n``, ``c_n``, ``T_t`` or ``k``.  ``[ASSUMPTION]`` — our choices:

* ``a_n = a0 / (n+1)^alpha``, ``c_n = c0 / (n+1)^gamma`` with the classical KW exponents
  ``alpha = 0.602``, ``gamma = 0.101`` (Spall's recommended values; the paper says nothing).
* ``mode="coordinate"`` (default): one *iteration* is a sweep over all coordinates in random
  order; for each coordinate the central difference ``±c_n`` is evaluated, the KW step
  ``x_i − a_n ĝ_i`` is proposed, and the best of the three probes is put to the Metropolis rule.
  One such iteration costs ``3·n_params`` loss evaluations (108 for the Fig-5 network); with it
  the bit-count KL reaches ≈1e-3 in ~10 sweeps ≈ 1000 evaluations.  The paper's "iteration" unit
  is not defined, so Fig 6 is reported against both sweeps and ``evaluations``
  (``reviews/001`` #8).  Whole-vector KW steps (``mode="full"``) stalled near KL ≈ 1e-2 at every
  setting tried, even at 7000+ evaluations.
* ``mode="full"``: the candidate is ``x̂ = x − a_n ĝ + jitter·T·N(0, 1)``, a descent step on the
  full central-difference gradient plus optional Gaussian exploration.
* geometric cooling ``T_{t+1} = cooling · T_t`` from ``t0``; ``k`` is the damping factor of Eq 3.
* the loss is treated as "minimise" (the paper's ``+a_n`` is for maximisation of ``M``).

With ``t0 = 0`` and ``jitter = 0`` this is plain KW descent with greedy acceptance.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

Params = npt.NDArray[np.float64]
Loss = Callable[[Params], float]
MapFn = Callable[[Loss, Sequence[Params]], Sequence[float]]


@dataclass(frozen=True)
class KWSAConfig:
    iterations: int = 100
    a0: float = 0.5
    alpha: float = 0.602
    c0: float = 0.1
    gamma: float = 0.101
    t0: float = 0.0
    cooling: float = 0.95
    k: float = 1.0
    jitter: float = 0.0
    seed: int = 0
    mode: Literal["coordinate", "full"] = "coordinate"
    restate_incumbent: bool = False
    """Re-evaluate the incumbent loss at the start of every iteration.  Needed when the objective
    is non-stationary (mini-batches, rotating episode seeds): a stale incumbent corrupts the
    Metropolis comparison — observed as 20/25 → 0/25 perfect Flappy games when training seeds
    were rotated without this flag, and 25/25 with it (``experiments/runs/flappy/batch5``)."""


@dataclass
class KWSAResult:
    """``params``/``loss`` are the best point seen (use these); ``final_params``/``final_loss``
    are the annealer's last accepted state (resume from these); ``history[i]`` is the accepted
    loss after iteration ``i`` (``history[0]`` = start) and ``best_history`` its running minimum."""

    params: Params
    loss: float
    history: list[float] = field(default_factory=list)
    accepted: int = 0
    evaluations: int = 0
    final_params: Params | None = None
    final_loss: float = float("nan")
    best_history: list[float] = field(default_factory=list)


def accept(delta: float, temperature: float, k: float, rng: np.random.Generator) -> bool:
    """Metropolis rule of ``[PTQNN Eq 3]``: always for ``Δe ≤ 0``, else with ``exp(−Δe/kT)``."""
    if delta <= 0:
        return True
    if temperature <= 0:
        return False
    return bool(rng.random() < np.exp(-delta / (k * temperature)))


def _serial_map(f: Loss, xs: Sequence[Params]) -> Sequence[float]:
    return [f(x) for x in xs]


def kw_gradient(f: Loss, x: Params, c: float, map_fn: MapFn = _serial_map) -> tuple[Params, int]:
    """Central-difference gradient, one ``±c`` pair per coordinate; returns (ĝ, #evaluations).
    The ``2n`` probes are independent and go through ``map_fn`` (e.g. a process pool)."""
    probes: list[Params] = []
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = c
        probes += [x + e, x - e]
    vals = np.asarray(map_fn(f, probes), dtype=np.float64)
    g = (vals[0::2] - vals[1::2]) / (2 * c)
    return g, 2 * x.size


def minimise(
    f: Loss,
    x0: Params,
    cfg: KWSAConfig,
    callback: Callable[[int, Params, float], None] | None = None,
    start_iteration: int = 0,
    map_fn: MapFn = _serial_map,
) -> KWSAResult:
    """Run ``cfg.iterations`` KW–SA steps from ``x0``; the schedule index starts at
    ``start_iteration`` so callers can resume or interleave runs.  Returns the best point seen."""
    rng = np.random.default_rng(cfg.seed)
    x = np.asarray(x0, dtype=np.float64).copy()
    fx = f(x)
    best_x, best_f = x.copy(), fx
    res = KWSAResult(params=x, loss=fx, history=[fx], evaluations=1, best_history=[fx])
    temperature = cfg.t0 * cfg.cooling**start_iteration
    for n in range(start_iteration, start_iteration + cfg.iterations):
        a_n = cfg.a0 / (n + 1) ** cfg.alpha
        c_n = cfg.c0 / (n + 1) ** cfg.gamma
        if cfg.restate_incumbent:
            fx = f(x)
            res.evaluations += 1
            if fx < best_f:
                best_x, best_f = x.copy(), fx
        if cfg.mode == "coordinate":
            for i in rng.permutation(x.size):
                e = np.zeros_like(x)
                e[i] = c_n
                fp, fm = map_fn(f, [x + e, x - e])
                step = np.zeros_like(x)
                step[i] = -a_n * (fp - fm) / (2 * c_n)
                fs = f(x + step)
                res.evaluations += 3
                fc, cand = min((fp, x + e), (fm, x - e), (fs, x + step), key=lambda t: t[0])
                if accept(fc - fx, temperature, cfg.k, rng):
                    x, fx = cand, fc
                    res.accepted += 1
                    if fx < best_f:
                        best_x, best_f = x.copy(), fx
        else:
            g, ev = kw_gradient(f, x, c_n, map_fn)
            res.evaluations += ev
            cand = x - a_n * g
            if cfg.jitter > 0:
                cand = cand + cfg.jitter * temperature * rng.normal(size=x.size)
            fc = f(cand)
            res.evaluations += 1
            if accept(fc - fx, temperature, cfg.k, rng):
                x, fx = cand, fc
                res.accepted += 1
                if fx < best_f:
                    best_x, best_f = x.copy(), fx
        temperature *= cfg.cooling
        res.history.append(fx)
        res.best_history.append(best_f)
        if callback is not None:
            callback(n, x, fx)
    res.params, res.loss = best_x, best_f
    res.final_params, res.final_loss = x, fx
    return res
