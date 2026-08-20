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
* the candidate is ``x̂ = x − a_n ĝ + jitter·N(0, 1)``: a descent step on the central-difference
  gradient ``ĝ`` (one pair of evaluations per coordinate) plus optional Gaussian exploration.
* geometric cooling ``T_{t+1} = cooling · T_t`` from ``t0``; ``k`` is the damping factor of Eq 3.
* the loss is treated as "minimise" (the paper's ``+a_n`` is for maximisation of ``M``).

With ``t0 = 0`` and ``jitter = 0`` this is plain KW descent with greedy acceptance.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

Params = npt.NDArray[np.float64]
Loss = Callable[[Params], float]


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


@dataclass
class KWSAResult:
    params: Params
    loss: float
    history: list[float] = field(default_factory=list)
    accepted: int = 0
    evaluations: int = 0


def accept(delta: float, temperature: float, k: float, rng: np.random.Generator) -> bool:
    """Metropolis rule of ``[PTQNN Eq 3]``: always for ``Δe ≤ 0``, else with ``exp(−Δe/kT)``."""
    if delta <= 0:
        return True
    if temperature <= 0:
        return False
    return bool(rng.random() < np.exp(-delta / (k * temperature)))


def kw_gradient(f: Loss, x: Params, c: float) -> tuple[Params, int]:
    """Central-difference gradient, one ``±c`` pair per coordinate; returns (ĝ, #evaluations)."""
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = c
        g[i] = (f(x + e) - f(x - e)) / (2 * c)
    return g, 2 * x.size


def minimise(
    f: Loss,
    x0: Params,
    cfg: KWSAConfig,
    callback: Callable[[int, Params, float], None] | None = None,
) -> KWSAResult:
    rng = np.random.default_rng(cfg.seed)
    x = np.asarray(x0, dtype=np.float64).copy()
    fx = f(x)
    best_x, best_f = x.copy(), fx
    res = KWSAResult(params=x, loss=fx, history=[fx], evaluations=1)
    temperature = cfg.t0
    for n in range(cfg.iterations):
        a_n = cfg.a0 / (n + 1) ** cfg.alpha
        c_n = cfg.c0 / (n + 1) ** cfg.gamma
        g, ev = kw_gradient(f, x, c_n)
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
        if callback is not None:
            callback(n, x, fx)
    res.params, res.loss = best_x, best_f
    return res
