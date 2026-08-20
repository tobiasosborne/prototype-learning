"""The quantum agent of ``[PTQNN Fig 10]``: observation → RX angles → network → flap decision.

"the position of the bird, as well as the position of the tubes is translated to (input) qubit
states. Based on this input, which are encoded by using RX-Gates, a quantum neural network
decides, whether to fly upwards or downwards" ``[PTQNN §III.D]``.  Fig 10 shows the concrete
circuit for the trained agent: ``RX(1.9537)`` and ``RX(2.1962)`` on the two input qubits, four
prototype neurons on qubits 3–6, and a classic output neuron on qubit 7 with bias ``RX(-1.4277)``
and ``CRX(-1.0422, -0.9988, -1.2622, -0.91002)`` — i.e. exactly
``Network(n_inputs=2, n_hidden=4, hidden="prototype")``.

Angle map: ``θ = 2·asin(√x)``, so that the prepared input qubit carries the feature *as a
probability*, ``P(qubit = 1) = sin²(θ/2) = x`` ``[SNN Fig 4, §III.A]``: "To translate this
probability score to an appropriate rotation we simply use asin(√aᵢ) as angular measure for the
rotation", pinned numerically by Fig 4's printed ``CRX(1.369) = 2·asin(√0.4)`` and
``CRX(1.7722) = 2·asin(√0.6)`` (see ``docs/physics/rosenhahn_osborne_hirche_snn.md §3.3``).
``mapping="linear"`` (``θ = π·x``) is kept for ablation — it is the naive reading of the ECG
figure ``[PTQNN Fig 12]`` and the two maps agree exactly at ``x = 0`` and ``x = 0.5, 1``.
Fig 10's own input angles read as ``x = sin²(1.9537/2) = 0.686`` and ``sin²(2.1962/2) = 0.792``
under the pinned map (a bird below mid-screen with the gap below it); ``[ASSUMPTION]`` remains
which *feature* each qubit carries.  ``angle_scale`` multiplies the resulting angle.

Decision rule ``[ASSUMPTION]``: flap iff ``P(out = 1) > threshold`` with ``threshold = 0.5``.
The paper only says the network "decides, whether to fly upwards or downwards"; the output qubit
of the network carries an activation probability ``[PTQNN Fig 4]``, and the obvious reading of a
binary decision from it is a median split.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from ptqnn.backends import reference as ref
from ptqnn.backends.base import Backend
from ptqnn.circuits import ir
from ptqnn.circuits.neurons import Network
from ptqnn.tasks.flappy.env import FlappyConfig, FlappyEnv, FlappyState

Real = npt.NDArray[np.float64]
Complex = npt.NDArray[np.complex128]


ANGLE_MAPS: dict[str, Callable[[Real], Real]] = {
    "amplitude": lambda x: 2.0 * np.arcsin(np.sqrt(np.clip(x, 0.0, 1.0))),
    "linear": lambda x: np.pi * np.asarray(x, dtype=np.float64),
}
"""``"amplitude"``: ``θ = 2·asin(√x)`` so that ``P(qubit = 1) = x`` ``[SNN Fig 4, §III.A]`` —
the default.  ``"linear"``: ``θ = π·x``, the ablation."""


def angles(obs: Sequence[float] | Real, mapping: str = "amplitude", scale: float = 1.0) -> Real:
    """Feature vector → RX angles."""
    return scale * ANGLE_MAPS[mapping](np.asarray(obs, dtype=np.float64))


def encode(
    obs: Sequence[float] | Real, mapping: str = "amplitude", scale: float = 1.0
) -> list[ir.Op]:
    """Feature vector → input-qubit preparation ``[RX(θ_i) on qubit i]``.

    ``[PTQNN §III.D]``: the inputs "are encoded by using RX-Gates".  Qubit ``i`` is the ``i``-th
    input qubit of :class:`~ptqnn.circuits.neurons.Network` (qubit index ``i``); the angle map is
    ``θ = 2·asin(√x)`` ``[SNN Fig 4, §III.A]`` unless ``mapping="linear"``.
    """
    return [ir.rx(i, float(t)) for i, t in enumerate(angles(obs, mapping, scale))]


class Policy(Protocol):
    """Anything that turns an observation into a flap decision."""

    def act(self, obs: Real) -> bool: ...


def quantise(obs: Real, levels: int) -> tuple[float, ...]:
    """Snap features to a ``1/levels`` grid; ``levels <= 0`` means "no quantisation" (the raw
    feature values become the cache key, which only ever hits on an exactly repeated state)."""
    if levels <= 0:
        return tuple(float(x) for x in obs)
    return tuple(float(round(float(x) * levels)) for x in obs)


@dataclass
class QuantumFlappyPolicy:
    """``P(out = 1)`` of the prototype network as a flap probability, thresholded.

    A game of 5000 frames would need 5000 seven-qubit simulations; almost all of them repeat a
    state the agent has already seen, so decisions are memoised on the *quantised* observation
    (``cache_levels`` steps per feature, 64 by default).  The policy is therefore literally a
    step function on a ``65 × 65`` grid — ``[ASSUMPTION]``; set ``cache_levels=0`` for the
    unquantised policy (much slower, identical behaviour up to the grid resolution).

    ``fast_linear`` exploits that the observation enters only through the two input rotations:
    the prepared state lives in the ``2^n_inputs``-dimensional span of ``|b, 0…0⟩`` and the rest
    of the circuit is a fixed unitary, so the network is evaluated once per parameter vector on
    that basis and every later observation costs one small matrix–vector product.  It is exact
    for pure-state (noiseless) backends only; with a channel-carrying backend, or with
    ``fast_linear=False``, every call goes through
    :func:`ptqnn.train.distribution.output_activation` instead.
    """

    net: Network
    params: Real
    backend: Backend = field(default_factory=ref.StatevectorBackend)
    threshold: float = 0.5
    angle_map: str = "amplitude"
    angle_scale: float = 1.0
    cache_levels: int = 64
    fast_linear: bool = True
    decision_cache: dict[tuple[float, ...], bool] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.params = np.asarray(self.params, dtype=np.float64).ravel()
        self._columns: Complex | None = None
        if self.fast_linear and isinstance(self.backend, ref.StatevectorBackend):
            self._columns = self._input_basis_columns()

    # -- evaluation -------------------------------------------------------------------------

    def _input_basis_columns(self) -> Complex:
        """``V[:, b] = U_tail |b, 0…0⟩`` for the ``2^n_inputs`` input basis strings."""
        net, n = self.net, self.net.n_qubits
        tail = net.circuit(self.params)
        cols = np.empty((2**n, 2**net.n_inputs), dtype=np.complex128)
        for b in range(2**net.n_inputs):
            psi0 = np.zeros(2**n, dtype=np.complex128)
            # qubit q contributes bit (n-1-q) of the basis index; inputs are qubits 0..n_in-1.
            idx = sum(((b >> (net.n_inputs - 1 - q)) & 1) << (n - 1 - q) for q in net.input_qubits)
            psi0[idx] = 1.0
            cols[:, b] = ref.run_statevector(tail, psi0)
        return cols

    def _fast_activation(self, obs: Real, columns: Complex) -> float:
        net = self.net
        # RX(θ)|0⟩ = cos(θ/2)|0⟩ − i sin(θ/2)|1⟩; the prepared input state is the product.
        half = 0.5 * angles(obs, self.angle_map, self.angle_scale)
        amp = np.stack([np.cos(half), -1j * np.sin(half)])  # (2, n_inputs)
        c = np.ones(1, dtype=np.complex128)
        for i in range(net.n_inputs):
            c = np.kron(c, amp[:, i])
        psi = columns @ c
        p = np.abs(psi) ** 2
        bit = net.n_qubits - 1 - net.output_qubit
        mask = (np.arange(p.size) >> bit) & 1 == 1
        return float(p[mask].sum())

    def activation(self, obs: Real) -> float:
        """``P(out = 1)`` for this observation."""
        if self._columns is not None:
            return self._fast_activation(obs, self._columns)
        from ptqnn.train import distribution

        prep = encode(obs, self.angle_map, self.angle_scale)
        return distribution.output_activation(self.net, self.params, prep, self.backend)

    def act(self, obs: Real) -> bool:
        """Flap iff ``P(out = 1) > threshold`` ``[ASSUMPTION]``."""
        key = quantise(obs, self.cache_levels)
        hit = self.decision_cache.get(key)
        if hit is not None:
            return hit
        grid = np.asarray(obs, dtype=np.float64)
        if self.cache_levels > 0:
            grid = np.asarray(key, dtype=np.float64) / self.cache_levels
        decision = bool(self.activation(grid) > self.threshold)
        self.decision_cache[key] = decision
        return decision

    def decision_grid(self, levels: int = 64) -> npt.NDArray[np.bool_]:
        """``act`` evaluated on the full 2-feature grid — the policy's decision boundary."""
        if self.net.n_inputs != 2:
            raise ValueError("decision_grid is only defined for two features")
        xs = np.linspace(0.0, 1.0, levels)
        return np.array([[self.act(np.array([a, b])) for b in xs] for a in xs], dtype=bool)


@dataclass
class ThresholdPolicy:
    """Classical bang-bang baseline: flap iff the bird is below the gap centre.

    Exists to prove the clone is solvable at all and to sanity-check the observation design —
    if *this* cannot play, the two features do not carry enough information and no network can.
    """

    margin: float = 0.0

    def act(self, obs: Real) -> bool:
        return bool(obs[0] > obs[1] + self.margin)


@dataclass
class EpisodeResult:
    """Outcome of one game."""

    score: int
    """Pipes passed."""
    steps: int
    total_reward: float
    tracking_error: float
    """Mean over the episode of ``|bird height − gap height|`` in normalised units — a dense,
    always-defined progress signal used for reward shaping in
    :mod:`ptqnn.tasks.flappy.train` ``[ASSUMPTION]``; the paper says nothing about the reward."""
    died: bool
    trajectory: tuple[FlappyState, ...] = ()

    @property
    def survived(self) -> bool:
        return not self.died


def play_episode(
    env: FlappyEnv,
    policy: Policy,
    max_steps: int = 5000,
    record: bool = False,
    on_step: Callable[[FlappyState, bool], None] | None = None,
) -> EpisodeResult:
    """Reset ``env`` and play until death or ``max_steps`` frames.

    ``record=True`` keeps a snapshot per frame (for :func:`render_episode`); it costs memory
    proportional to ``max_steps`` and is off during training.
    """
    obs = env.reset()
    frames: list[FlappyState] = [env.snapshot()] if record else []
    total, errors, died = 0.0, [abs(float(obs[0]) - float(obs[1]))], False
    for _ in range(max_steps):
        flap = policy.act(obs)
        if on_step is not None:
            on_step(env.snapshot(), flap)
        obs, reward, done, _ = env.step(flap)
        total += reward
        errors.append(abs(float(obs[0]) - float(obs[1])))
        if record:
            frames.append(env.snapshot())
        if done:
            died = True
            break
    return EpisodeResult(
        score=env.score,
        steps=env.steps,
        total_reward=total,
        tracking_error=float(np.mean(errors)),
        died=died,
        trajectory=tuple(frames),
    )


# --- rendering ------------------------------------------------------------------------------

_SKY = (78, 192, 202)
_GROUND = (222, 216, 149)
_GRASS = (117, 195, 78)
_PIPE = (87, 201, 80)
_PIPE_EDGE = (60, 130, 55)
_BIRD = (250, 210, 60)
_BIRD_EYE = (40, 40, 40)


def _fill(
    img: npt.NDArray[np.uint8],
    y0: float,
    y1: float,
    x0: float,
    x1: float,
    rgb: tuple[int, int, int],
) -> None:
    h, w = img.shape[:2]
    a, b = max(0, round(y0)), min(h, round(y1))
    c, d = max(0, round(x0)), min(w, round(x1))
    if a < b and c < d:
        img[a:b, c:d] = rgb


def render_state(cfg: FlappyConfig, state: FlappyState, scale: int = 1) -> npt.NDArray[np.uint8]:
    """Rasterise one frame as an ``(H, W, 3)`` uint8 RGB array — pure numpy, no pygame.

    Mirrors ``[PTQNN Fig 11]``: sky, two tubes per pipe pair with the gap between them, a ground
    strip with a grass edge, and the bird.  The score is drawn as a tick bar in the top-left
    corner (there is no font in numpy).
    """
    h, w = int(cfg.screen_height), int(cfg.screen_width)
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:] = _SKY
    for p in state.pipes:
        for y0, y1 in ((0.0, p.gap_top(cfg)), (p.gap_bottom(cfg), cfg.playable_height)):
            _fill(img, y0, y1, p.x, p.x + cfg.pipe_width, _PIPE)
            _fill(img, y0, y1, p.x, p.x + 3, _PIPE_EDGE)
            _fill(img, y0, y1, p.x + cfg.pipe_width - 3, p.x + cfg.pipe_width, _PIPE_EDGE)
        _fill(img, p.gap_top(cfg) - 12, p.gap_top(cfg), p.x - 4, p.x + cfg.pipe_width + 4, _PIPE)
        _fill(
            img, p.gap_bottom(cfg), p.gap_bottom(cfg) + 12, p.x - 4, p.x + cfg.pipe_width + 4, _PIPE
        )
    _fill(img, cfg.playable_height, cfg.playable_height + 10, 0, w, _GRASS)
    _fill(img, cfg.playable_height + 10, h, 0, w, _GROUND)
    bx, by = cfg.bird_x, state.bird_y
    _fill(
        img,
        by - 0.5 * cfg.bird_height,
        by + 0.5 * cfg.bird_height,
        bx - 0.5 * cfg.bird_width,
        bx + 0.5 * cfg.bird_width,
        _BIRD,
    )
    _fill(img, by - 8, by - 3, bx + 6, bx + 11, _BIRD_EYE)
    for i in range(min(state.score, (w - 12) // 5)):
        _fill(img, 6, 14, 6 + 5 * i, 9 + 5 * i, (255, 255, 255))
    if scale > 1:
        img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
    return img


def render_frame(env: FlappyEnv, scale: int = 1) -> npt.NDArray[np.uint8]:
    """Rasterise the environment's current state."""
    return render_state(env.config, env.snapshot(), scale)


def render_episode(
    cfg: FlappyConfig,
    trajectory: Sequence[FlappyState],
    max_frames: int | None = None,
    scale: int = 1,
) -> list[npt.NDArray[np.uint8]]:
    frames = trajectory if max_frames is None else trajectory[:max_frames]
    return [render_state(cfg, s, scale) for s in frames]


def save_gif(frames: Sequence[npt.NDArray[np.uint8]], path: str | Path, fps: int = 30) -> Path:
    """Write an animated GIF.  Needs Pillow or matplotlib — both optional dependencies."""
    if not frames:
        raise ValueError("no frames to save")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except ImportError:
        pass
    else:
        images = [Image.fromarray(np.asarray(f, dtype=np.uint8)) for f in frames]
        images[0].save(
            out,
            save_all=True,
            append_images=images[1:],
            duration=max(1, round(1000 / fps)),
            loop=0,
        )
        return out
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - both backends missing
        raise ImportError("save_gif needs Pillow or matplotlib installed") from exc
    fig, ax = plt.subplots(figsize=(frames[0].shape[1] / 100, frames[0].shape[0] / 100), dpi=100)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    im = ax.imshow(frames[0])

    def _update(i: int) -> list[Any]:
        im.set_data(frames[i])
        return [im]

    anim = animation.FuncAnimation(fig, _update, frames=len(frames), interval=1000 / fps)
    anim.save(out, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return out
