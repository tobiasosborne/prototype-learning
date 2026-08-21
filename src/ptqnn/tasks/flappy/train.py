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
  is a non-stationary objective, so ``train(batch > 0)`` switches on
  ``KWSAConfig.restate_incumbent`` (the incumbent is re-scored on every new batch; without it the
  Metropolis comparison used a stale loss and held-out play collapsed from 20/25 to 0/25 perfect
  games — review 002 #8) and reports the annealer's *final* point rather than the best-ever one,
  which under a rotating batch is merely the point that drew the luckiest batch.  Measured:
  25/25 perfect held-out games with a 5-of-10 rotating batch (``experiments/runs/flappy/batch5``)
  vs 25/25 with fixed seeds (``…/batch0``) — no longer worse, not better either.  Off by default
  because it buys nothing here.
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
    margin: float = 0.0
    """Margin-aware training (shot-noise programme, item 1): frames whose decision margin
    ``δ = |P(out=1) − ½|`` is below ``margin`` are penalised by ``margin_weight · hinge``, with
    ``hinge = mean_frames max(0, 1 − δ/margin)``.  Shots to decide a frame scale as ``1/δ²``, so
    lifting the median margin from 0.02 to 0.1 is ×25 fewer shots; it costs nothing noiselessly.
    ``margin = 0`` (default) reproduces the paper-style objective.  ``[ASSUMPTION]``."""
    margin_weight: float = 0.0


def margin_hinge(margins: Sequence[float], margin: float) -> float:
    """``mean_i max(0, 1 − δ_i/m)`` over the played frames — 0 when every frame clears ``m``,
    1 when every frame sits exactly on the boundary."""
    if margin <= 0 or len(margins) == 0:
        return 0.0
    d = np.asarray(margins, dtype=np.float64)
    return float(np.mean(np.maximum(0.0, 1.0 - d / margin)))


def make_policy(
    net: Network,
    params: Real,
    cfg: ObjectiveConfig,
    backend: Backend | None = None,
    shots: int = 0,
    shot_seed: int = 0,
    decision: ag.SPRTDecision | None = None,
) -> ag.QuantumFlappyPolicy:
    return ag.QuantumFlappyPolicy(
        net=net,
        params=np.asarray(params, dtype=np.float64),
        backend=backend or ref.StatevectorBackend(),
        threshold=cfg.threshold,
        angle_map=cfg.angle_map,
        angle_scale=cfg.angle_scale,
        cache_levels=cfg.cache_levels,
        shots=shots,
        shot_seed=shot_seed,
        decision=decision,
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
    objective function from noisy function evaluations" ``[PTQNN §II.C]``).  Across iterations
    the objective changes, so :func:`train` pairs it with ``restate_incumbent=True`` — without
    that flag :func:`ptqnn.train.kwsa.minimise` reused the incumbent loss from the previous batch
    and held-out play collapsed from 20/25 to 0/25 perfect games.
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
        value = episode_loss(results, c.shaping_weight)
        if c.margin > 0 and c.margin_weight > 0:
            value += c.margin_weight * margin_hinge(policy.frame_margins, c.margin)
        return value

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
    :class:`SeedRotation` batch; the optimiser then re-scores the incumbent every iteration
    (``restate_incumbent``) and the returned parameters are the annealer's final state, not its
    best-ever one (see the module docstring).  ``eval_seeds`` are the held-out episodes the
    final score is reported on.
    """
    k = cfg or DEFAULT_KWSA
    oc = replace(objective_config or ObjectiveConfig(), max_steps=max_steps)
    rng = np.random.default_rng(k.seed)

    rotating = 0 < batch < len(seeds)
    train_seeds: Sequence[int] | SeedRotation = (
        SeedRotation(seeds, batch, rng) if rotating else seeds
    )
    if rotating and not k.restate_incumbent:
        k = replace(k, restate_incumbent=True)

    cheap = objective(net, train_seeds, replace(oc, max_steps=restart_steps), backend)
    x0, f0 = random_restarts(net, cheap, n_restarts, rng)

    def on_iteration(n: int, x: Real, fx: float) -> None:
        if callback is not None:
            callback(n, x, fx)
        if isinstance(train_seeds, SeedRotation):
            train_seeds.advance()

    loss = objective(net, train_seeds, oc, backend)
    res = kwsa.minimise(loss, x0, k, callback=on_iteration)
    # Under a rotating batch "best loss ever seen" selects the luckiest batch; the final
    # accepted state is the honest answer (review 002 #8).
    params = res.final_params if (rotating and res.final_params is not None) else res.params
    loss_out = res.final_loss if rotating else res.loss

    policy = make_policy(net, params, oc, backend)
    results = evaluate(policy, eval_seeds or seeds, eval_steps, oc.env_config, oc.features)
    return FlappyTrainResult(
        params=params,
        loss=loss_out,
        history=res.history,
        kwsa=res,
        init_loss=f0,
        evaluations=res.evaluations + n_restarts,
        scores=[r.score for r in results],
    )
