"""Training the Flappy agent: episode return → KW–SA ``[PTQNN §II.C, §III.D]``.

The paper says only that the trained prototype network with four hidden neurons "plays the game
perfectly, we stopped a run after the score of 450.000 which took 6 days" ``[PTQNN §III.D]``.
No reward, no episode protocol and no optimiser schedule are published, so the whole objective
below is ``[ASSUMPTION]``:

* **Loss** = ``−(mean episode return over a few seeded episodes) + λ · (mean tracking error)``.
  The return is ``0.01`` per frame survived plus ``1`` per pipe; the shaping term is the mean
  ``|bird height − gap height|`` in normalised units.  Shaping is needed because the raw return
  is nearly binary on this clone — a policy either dies within ~30 frames or survives
  indefinitely (see ``tests/test_flappy.py``), so finite differences of the return alone are
  zero almost everywhere.  The tracking error varies smoothly across the dead region and points
  at the surviving one.
* **Multi-start.** ``n_restarts`` random parameter vectors are scored on short episodes and the
  best one seeds the KW–SA run.  17 parameters is small enough that random search finds a decent
  basin cheaply, and KW–SA then only has to polish it.
* **Episode length over episode count.** Coverage of pipe-height *transitions* is what
  generalises: five 1500-frame episodes (≈165 transitions) beat eight 800-frame ones on
  held-out seeds (20/20 vs 15/20 perfect games).  A rotating seed batch (:class:`SeedRotation`)
  looks like the textbook fix and is *worse* here — 0/25 — because
  :func:`ptqnn.train.kwsa.minimise` carries its incumbent loss across iterations without
  re-evaluating it, so a rotated batch makes the Metropolis comparison meaningless.  It is kept
  off by default (``batch=0``) and would need an optimiser that re-scores the incumbent.
* **Budget.** We do not attempt the paper's 6-day run.  The target is deterministic perfect play
  on our clone under a step cap: 5000 frames at one pipe per 45 frames is 110 pipes.  The
  defaults (300 restarts + 25 coordinate sweeps ≈ 1575 loss evaluations, ~3 min) reach 110/110
  on all 20 held-out seeds, for every optimiser seed tried.

Everything is single-process and deterministic: with the fast linear evaluation path and the
decision cache of :class:`~ptqnn.tasks.flappy.agent.QuantumFlappyPolicy` a 1500-frame episode
costs ≈45 ms (dominated by the clone, not the simulator), so a full run is a couple of minutes
and multiprocessing would only add nondeterminism.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt

from ptqnn.backends import reference as ref
from ptqnn.backends.base import Backend
from ptqnn.circuits.neurons import Network
from ptqnn.tasks.flappy import agent as ag
from ptqnn.tasks.flappy.env import FlappyConfig, FlappyEnv
from ptqnn.train import kwsa

Real = npt.NDArray[np.float64]
Loss = Callable[[Real], float]

DEFAULT_KWSA = kwsa.KWSAConfig(
    iterations=25,
    a0=0.4,
    alpha=0.602,
    c0=0.25,
    gamma=0.101,
    t0=0.05,
    cooling=0.9,
    k=1.0,
    seed=0,
    mode="coordinate",
)
"""``[ASSUMPTION]`` — no schedule is published.  Coordinate sweeps (17 parameters × 3 probes =
51 episode-batch evaluations per iteration, so 25 sweeps ≈ 1275 evaluations), a comparatively
large difference step ``c0 = 0.25`` rad because the loss is a step function of the decision
boundary, and a small starting temperature so a sweep can climb out of a plateau.  The budget is
set by *robustness*, not by the median run: 12 sweeps after 200 restarts already reaches 110/110
on 20 held-out seeds for most optimiser seeds, but fails outright (0/20, loss ≈ −2.8) for one in
four — the restarts never find a surviving basin and 12 sweeps cannot climb out.  300 restarts +
25 sweeps was perfect on every optimiser seed we tried."""


@dataclass(frozen=True)
class ObjectiveConfig:
    """What one loss evaluation costs and what it measures.  ``[ASSUMPTION]`` throughout."""

    env_config: FlappyConfig = field(default_factory=FlappyConfig)
    features: str = "bird_gap"
    max_steps: int = 800
    """Step cap per training episode (≈ 17 pipes at the default spacing)."""
    shaping_weight: float = 1.0
    threshold: float = 0.5
    angle_map: str = "amplitude"
    """``θ = 2·asin(√x)`` ``[SNN Fig 4, §III.A]``; ``"linear"`` for the ``θ = π·x`` ablation."""
    angle_scale: float = 1.0
    cache_levels: int = 64


def make_policy(
    net: Network, params: Real, cfg: ObjectiveConfig, backend: Backend | None = None
) -> ag.QuantumFlappyPolicy:
    return ag.QuantumFlappyPolicy(
        net=net,
        params=np.asarray(params, dtype=np.float64),
        backend=backend or ref.StatevectorBackend(),
        threshold=cfg.threshold,
        angle_map=cfg.angle_map,
        angle_scale=cfg.angle_scale,
        cache_levels=cfg.cache_levels,
    )


def evaluate(
    policy: ag.Policy,
    seeds: Sequence[int],
    max_steps: int,
    env_config: FlappyConfig | None = None,
    features: str = "bird_gap",
    record: bool = False,
) -> list[ag.EpisodeResult]:
    """Play one episode per seed."""
    cfg = env_config or FlappyConfig()
    return [
        ag.play_episode(
            FlappyEnv(config=cfg, seed=s, features=features), policy, max_steps, record=record
        )
        for s in seeds
    ]


def episode_loss(results: Sequence[ag.EpisodeResult], shaping_weight: float) -> float:
    """``−mean return + λ · mean tracking error`` — the number KW–SA minimises."""
    ret = float(np.mean([r.total_reward for r in results]))
    err = float(np.mean([r.tracking_error for r in results]))
    return -ret + shaping_weight * err


class SeedRotation:
    """A rotating mini-batch of episode seeds — **off by default**, see the module docstring.

    The batch is held constant *within* an iteration so that that sweep's central differences
    see the same episodes; KW is built for noisy evaluations ("estimates the gradient of an
    objective function from noisy function evaluations" ``[PTQNN §II.C]``).  Across iterations,
    however, :func:`ptqnn.train.kwsa.minimise` reuses the incumbent loss it computed on the
    previous batch, so every rotation corrupts the Metropolis comparison; measured, this turns
    20/25 perfect held-out games into 0/25.  Kept for the record and for optimisers that
    re-score the incumbent.
    """

    def __init__(self, pool: Sequence[int], batch: int, rng: np.random.Generator) -> None:
        self.pool = tuple(pool)
        self.batch = min(batch, len(self.pool))
        self.rng = rng
        self.seeds: tuple[int, ...] = tuple(self.pool[: self.batch])
        self.advance()

    def advance(self) -> None:
        """Draw the next batch (called once per KW–SA iteration)."""
        idx = self.rng.choice(len(self.pool), size=self.batch, replace=False)
        self.seeds = tuple(self.pool[int(i)] for i in idx)


def objective(
    net: Network,
    seeds: Sequence[int] | SeedRotation = (0, 1, 2),
    cfg: ObjectiveConfig | None = None,
    backend: Backend | None = None,
) -> Loss:
    """Negative shaped episode return, averaged over ``seeds`` — the loss handed to
    :func:`ptqnn.train.kwsa.minimise`.  ``seeds`` may be a :class:`SeedRotation`, in which case
    each iteration is scored on a different batch of episodes."""
    c = cfg or ObjectiveConfig()

    def loss(params: Real) -> float:
        batch = seeds.seeds if isinstance(seeds, SeedRotation) else seeds
        policy = make_policy(net, params, c, backend)
        results = evaluate(policy, batch, c.max_steps, c.env_config, c.features)
        return episode_loss(results, c.shaping_weight)

    return loss


# --- classical baseline ---------------------------------------------------------------------


def baseline_policy(margin: float = 0.0) -> ag.ThresholdPolicy:
    """Flap iff the bird is below the gap centre — the cheap proof that the clone is solvable
    from the two paper features alone."""
    return ag.ThresholdPolicy(margin=margin)


# --- training -------------------------------------------------------------------------------


@dataclass
class FlappyTrainResult:
    params: Real
    loss: float
    history: list[float]
    kwsa: kwsa.KWSAResult
    init_loss: float
    evaluations: int
    scores: list[int] = field(default_factory=list)
    """Pipes passed per evaluation seed at the end of training (long step cap)."""

    @property
    def best_score(self) -> int:
        return max(self.scores) if self.scores else 0


def random_restarts(
    net: Network,
    loss: Loss,
    n_restarts: int,
    rng: np.random.Generator,
    scale: float = np.pi,
) -> tuple[Real, float]:
    """Cheapest possible global search: score ``n_restarts`` random parameter vectors, keep the
    best.  Returns ``(params, loss)``."""
    best_x = net.random_params(rng, scale)
    best_f = loss(best_x)
    for _ in range(max(0, n_restarts - 1)):
        x = net.random_params(rng, scale)
        f = loss(x)
        if f < best_f:
            best_x, best_f = x, f
    return best_x, best_f


def train(
    net: Network,
    cfg: kwsa.KWSAConfig | None = None,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    max_steps: int = 1500,
    objective_config: ObjectiveConfig | None = None,
    n_restarts: int = 300,
    restart_steps: int = 250,
    batch: int = 0,
    eval_seeds: Sequence[int] | None = None,
    eval_steps: int = 5000,
    backend: Backend | None = None,
    callback: Callable[[int, Real, float], None] | None = None,
) -> FlappyTrainResult:
    """Random multi-start, then KW–SA on the shaped episode return.

    ``seeds`` are the training episodes — every loss evaluation plays all of them.  With
    ``batch > 0`` they instead become a pool from which each iteration draws a fresh
    :class:`SeedRotation` batch (not recommended, see the module docstring).  ``eval_seeds`` are
    the held-out episodes the final score is reported on.
    """
    k = cfg or DEFAULT_KWSA
    oc = replace(objective_config or ObjectiveConfig(), max_steps=max_steps)
    rng = np.random.default_rng(k.seed)

    train_seeds: Sequence[int] | SeedRotation = (
        SeedRotation(seeds, batch, rng) if 0 < batch < len(seeds) else seeds
    )

    cheap = objective(net, train_seeds, replace(oc, max_steps=restart_steps), backend)
    x0, f0 = random_restarts(net, cheap, n_restarts, rng)

    def on_iteration(n: int, x: Real, fx: float) -> None:
        if callback is not None:
            callback(n, x, fx)
        if isinstance(train_seeds, SeedRotation):
            train_seeds.advance()

    loss = objective(net, train_seeds, oc, backend)
    res = kwsa.minimise(loss, x0, k, callback=on_iteration)

    policy = make_policy(net, res.params, oc, backend)
    results = evaluate(policy, eval_seeds or seeds, eval_steps, oc.env_config, oc.features)
    return FlappyTrainResult(
        params=res.params,
        loss=res.loss,
        history=res.history,
        kwsa=res,
        init_loss=f0,
        evaluations=res.evaluations + n_restarts,
        scores=[r.score for r in results],
    )
