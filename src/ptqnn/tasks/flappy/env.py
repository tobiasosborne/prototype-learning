"""A deterministic, headless Flappy Bird clone (numpy only, no pygame).

``[PTQNN §III.D]``: "Figure 10 shows the pipeline for training an agent to play Flappy Birds.
From the image, the position of the bird, as well as the position of the tubes is translated to
(input) qubit states. Based on this input, which are encoded by using RX-Gates, a quantum neural
network decides, whether to fly upwards or downwards. ... Our trained model plays the game
perfectly, we stopped a run after the score of 450.000 which took 6 days. ... As model we use
our prototype-architecture with 4 hidden neurons."

The paper publishes neither the game implementation nor its physics constants; ``[PTQNN Fig 10]``
only shows an "FPS: 60.17" overlay and two highlighted boxes (the bird's bounding box and the
gap between the two tubes) wired to the two input qubits, which are driven by ``RX(1.9537)`` and
``RX(2.1962)``.  Everything below is therefore ``[ASSUMPTION]``, listed in ``PLAN.md §0``:

* **Units.** Pixels and frames, one ``step()`` = one frame at a nominal 60 fps.  Constants are
  the classic *FlapPyBird* clone's (288x512 px screen, 112 px ground strip, 52 px pipe width,
  34x24 px bird, 4 px/frame scroll) with a softened flap (``-7`` instead of ``-9`` px/frame) and
  a wider gap/spacing (120/180 instead of 100/144 px).  Rationale: a *memoryless* policy sees
  only positions, never the velocity, so it can only be a bang-bang controller; its overshoot
  after a flap is ``impulse²/(2·gravity) ≈ 24.5`` px, which must fit inside the half-gap minus
  half the bird (``60 − 12 = 48`` px).  The classic constants (``40.5`` px overshoot against a
  38 px budget) make the game unsolvable *without* velocity, which would say nothing about the
  network.  The 3-feature variant (which adds horizontal distance, still no velocity) is offered
  for ablation.
* **Observation.**  Two features in ``[0, 1]`` — the bird's height and the vertical position of
  the next gap's centre, both divided by the playable height.  This is the reading of Fig 10's
  two boxes: bird box → qubit 1, gap box → qubit 2.  ``y`` grows *downwards* (screen
  convention), so "bird below the gap" is ``obs[0] > obs[1]``.
* **Reward.**  ``+1`` per pipe passed and ``+0.01`` per frame survived, nothing on death.  The
  per-frame term is what makes the episode return informative before the first pipe is reached.

Determinism: the gap heights are the *only* stochastic element and come from a seeded
``numpy.random.Generator``; two ``FlappyEnv``\\ s with the same seed and the same action sequence
produce bit-identical trajectories.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

Real = npt.NDArray[np.float64]


@dataclass(frozen=True)
class FlappyConfig:
    """Geometry, physics and reward of the clone.  All lengths in px, all times in frames.

    ``[ASSUMPTION]`` throughout — see the module docstring for the provenance of each number.
    """

    # --- geometry (px) ---
    screen_width: float = 288.0
    screen_height: float = 512.0
    ground_height: float = 112.0
    """Height of the non-playable ground strip at the bottom of the screen."""
    bird_x: float = 57.0
    """Horizontal position of the bird's *centre*; the bird never moves horizontally."""
    bird_width: float = 34.0
    bird_height: float = 24.0
    pipe_width: float = 52.0
    pipe_gap: float = 120.0
    """Vertical opening between the upper and the lower tube."""
    pipe_spacing: float = 180.0
    """Horizontal distance between consecutive pipe pairs."""
    gap_margin: float = 80.0
    """Minimum distance of a gap edge from the ceiling and from the ground; with the defaults
    the gap centre is uniform on ``[140, 260]`` px."""

    # --- physics ---
    gravity: float = 1.0
    """Downward acceleration, px/frame²."""
    flap_impulse: float = -7.0
    """Velocity the bird is *set* to by a flap (not added to), px/frame; negative is upwards."""
    max_fall_speed: float = 10.0
    """Terminal velocity, px/frame."""
    scroll_speed: float = 4.0
    """Pipe velocity, px/frame."""

    # --- reward ---
    step_reward: float = 0.01
    pipe_reward: float = 1.0
    death_reward: float = 0.0

    # --- episode start ---
    start_height_fraction: float = 0.5
    """Bird's initial height as a fraction of the playable height."""

    @property
    def playable_height(self) -> float:
        """Distance from the ceiling to the top of the ground strip (400 px by default)."""
        return self.screen_height - self.ground_height

    @property
    def frames_per_pipe(self) -> float:
        """Frames between two pipe centres passing the bird."""
        return self.pipe_spacing / self.scroll_speed

    @property
    def gap_centre_range(self) -> tuple[float, float]:
        lo = self.gap_margin + 0.5 * self.pipe_gap
        hi = self.playable_height - self.gap_margin - 0.5 * self.pipe_gap
        if not lo < hi:
            raise ValueError("gap_margin/pipe_gap leave no room for a gap centre")
        return lo, hi

    @property
    def flap_overshoot(self) -> float:
        """How far above the flap point a bang-bang controller coasts: ``v²/(2g)``."""
        return self.flap_impulse**2 / (2.0 * self.gravity)


@dataclass(frozen=True)
class Pipe:
    """One pipe *pair*; ``x`` is the left edge, ``gap_centre`` the middle of the opening."""

    x: float
    gap_centre: float
    passed: bool = False

    def gap_top(self, cfg: FlappyConfig) -> float:
        return self.gap_centre - 0.5 * cfg.pipe_gap

    def gap_bottom(self, cfg: FlappyConfig) -> float:
        return self.gap_centre + 0.5 * cfg.pipe_gap


@dataclass(frozen=True)
class FlappyState:
    """Everything needed to draw a frame or to compute an observation."""

    bird_y: float
    bird_vy: float
    pipes: tuple[Pipe, ...]
    score: int
    steps: int
    done: bool = False


# --- observation features -----------------------------------------------------------------


FeatureFn = Callable[[FlappyConfig, FlappyState], Real]
"""A pure map from (config, state) to features in ``[0, 1]``; one feature per input qubit."""


def next_pipe(cfg: FlappyConfig, state: FlappyState) -> Pipe | None:
    """The first pipe the bird has not cleared yet — the one Fig 10 highlights."""
    left = cfg.bird_x - 0.5 * cfg.bird_width
    for p in state.pipes:
        if p.x + cfg.pipe_width >= left:
            return p
    return None


def _unit(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _bird_and_gap(cfg: FlappyConfig, state: FlappyState) -> tuple[float, float]:
    h = cfg.playable_height
    p = next_pipe(cfg, state)
    gap = p.gap_centre if p is not None else 0.5 * h
    return _unit(state.bird_y / h), _unit(gap / h)


def features_bird_gap(cfg: FlappyConfig, state: FlappyState) -> Real:
    """``[ASSUMPTION]`` the paper's two inputs: bird height, next gap centre ``[PTQNN Fig 10]``."""
    bird, gap = _bird_and_gap(cfg, state)
    return np.array([bird, gap], dtype=np.float64)


def features_bird_gap_distance(cfg: FlappyConfig, state: FlappyState) -> Real:
    """Ablation variant: adds the horizontal distance to the next pipe's left edge, normalised
    by the screen width — ``0`` when the pipe is level with the bird, ``≈0.86`` at reset."""
    bird, gap = _bird_and_gap(cfg, state)
    p = next_pipe(cfg, state)
    left = cfg.bird_x - 0.5 * cfg.bird_width
    dx = 1.0 if p is None else (p.x - left) / cfg.screen_width
    return np.array([bird, gap, _unit(dx)], dtype=np.float64)


FEATURES: dict[str, FeatureFn] = {
    "bird_gap": features_bird_gap,
    "bird_gap_distance": features_bird_gap_distance,
}
"""Pluggable feature sets; ``FlappyEnv(features=...)`` takes a key or a :data:`FeatureFn`."""


def n_features(features: str | FeatureFn) -> int:
    """Number of input qubits a feature set needs (probed on a dummy state)."""
    fn = FEATURES[features] if isinstance(features, str) else features
    cfg = FlappyConfig()
    state = FlappyState(bird_y=0.5 * cfg.playable_height, bird_vy=0.0, pipes=(), score=0, steps=0)
    return int(fn(cfg, state).size)


# --- the environment ----------------------------------------------------------------------


@dataclass
class FlappyEnv:
    """Headless clone.  ``reset()`` → obs; ``step(flap)`` → (obs, reward, done, info).

    The episode ends when the bird touches the ceiling, the ground, or a tube.  ``score`` counts
    the pipe pairs whose right edge has moved past the bird's left edge.
    """

    config: FlappyConfig = field(default_factory=FlappyConfig)
    seed: int = 0
    features: str | FeatureFn = "bird_gap"
    n_pipes_ahead: int = 3
    """How many pipe pairs are kept in the state (the on-screen ones plus a spare)."""

    def __post_init__(self) -> None:
        self._feature_fn: FeatureFn = (
            FEATURES[self.features] if isinstance(self.features, str) else self.features
        )
        self.reset()

    # -- state ------------------------------------------------------------------------------

    @property
    def state(self) -> FlappyState:
        return FlappyState(
            bird_y=self._bird_y,
            bird_vy=self._bird_vy,
            pipes=tuple(self._pipes),
            score=self._score,
            steps=self._steps,
            done=self._done,
        )

    def snapshot(self) -> FlappyState:
        """Immutable copy of the state — safe to keep in a trajectory."""
        return self.state

    @property
    def score(self) -> int:
        """Pipes passed."""
        return self._score

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def done(self) -> bool:
        return self._done

    @property
    def n_features(self) -> int:
        return int(self.observation().size)

    # -- dynamics ---------------------------------------------------------------------------

    def _new_gap_centre(self) -> float:
        lo, hi = self.config.gap_centre_range
        return float(self._rng.uniform(lo, hi))

    def reset(self) -> Real:
        """Restart from the seed; returns the first observation."""
        cfg = self.config
        self._rng = np.random.default_rng(self.seed)
        self._bird_y = cfg.start_height_fraction * cfg.playable_height
        self._bird_vy = 0.0
        self._score = 0
        self._steps = 0
        self._done = False
        self._pipes: list[Pipe] = [
            Pipe(x=cfg.screen_width + i * cfg.pipe_spacing, gap_centre=self._new_gap_centre())
            for i in range(self.n_pipes_ahead)
        ]
        return self.observation()

    def observation(self) -> Real:
        """Pure function of the current state — no side effects, no RNG."""
        return self._feature_fn(self.config, self.state)

    def _collides(self) -> bool:
        cfg = self.config
        top, bottom = self._bird_y - 0.5 * cfg.bird_height, self._bird_y + 0.5 * cfg.bird_height
        if top <= 0.0 or bottom >= cfg.playable_height:
            return True
        left, right = cfg.bird_x - 0.5 * cfg.bird_width, cfg.bird_x + 0.5 * cfg.bird_width
        for p in self._pipes:
            overlaps = p.x < right and left < p.x + cfg.pipe_width
            if overlaps and (top < p.gap_top(cfg) or bottom > p.gap_bottom(cfg)):
                return True
        return False

    def step(self, flap: bool) -> tuple[Real, float, bool, dict[str, Any]]:
        """Advance one frame.  Stepping a finished episode raises."""
        if self._done:
            raise RuntimeError("episode is over; call reset()")
        cfg = self.config

        self._bird_vy = (
            cfg.flap_impulse if flap else min(self._bird_vy + cfg.gravity, cfg.max_fall_speed)
        )
        self._bird_y += self._bird_vy

        left = cfg.bird_x - 0.5 * cfg.bird_width
        passed = 0
        pipes: list[Pipe] = []
        for p in self._pipes:
            x, was_passed = p.x - cfg.scroll_speed, p.passed
            if not was_passed and x + cfg.pipe_width < left:
                was_passed, passed = True, passed + 1
            # retire pipes that have scrolled off the left edge
            if x + cfg.pipe_width > -cfg.bird_width:
                pipes.append(Pipe(x, p.gap_centre, was_passed))
        while len(pipes) < self.n_pipes_ahead:
            x = pipes[-1].x + cfg.pipe_spacing if pipes else cfg.screen_width
            pipes.append(Pipe(x=x, gap_centre=self._new_gap_centre()))
        self._pipes = pipes

        self._score += passed
        self._steps += 1
        self._done = self._collides()

        reward = (
            cfg.step_reward + passed * cfg.pipe_reward + (cfg.death_reward if self._done else 0.0)
        )
        info: dict[str, Any] = {"score": self._score, "steps": self._steps, "passed": passed}
        return self.observation(), float(reward), self._done, info

    def run(self, actions: Sequence[bool]) -> FlappyState:
        """Replay a fixed action sequence (test helper); stops early on death."""
        for a in actions:
            if self._done:
                break
            self.step(a)
        return self.state
