"""Bit counting ``[PTQNN §III.C, Fig 4–6]``.

"The task is bit-counting of a 5-dimensional bit string. The 5 bits are encoded as input ...
Prototype layers ... are connected to 6 hidden neurons. The outcome is then forwarded to one
output neuron. ... Strings with many 1-bits result in a higher activation probability and bit
strings with many 0 entries result in a lower probability."  Fig 5's circuit has 5 hidden
neurons (11 qubits); Fig 4's text says 6.  The target histogram in Fig 5 (bottom right) has
bar heights 0.009, 0.018, 0.027, 0.036, 0.045, 0.054 for popcount 0..5, i.e.
``p(x) = (popcount(x) + 1) / 112`` — the ``+1`` keeps ``00000`` representable (read from
figure, ±0.001).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from ptqnn.backends import reference as ref
from ptqnn.backends.base import Backend
from ptqnn.circuits import ir
from ptqnn.circuits.neurons import HiddenKind, Network
from ptqnn.train import distribution, kwsa, losses

Real = npt.NDArray[np.float64]
Loss = Callable[[Real], float]


def popcounts(n_bits: int) -> npt.NDArray[np.int64]:
    return np.array([bin(x).count("1") for x in range(2**n_bits)], dtype=np.int64)


def target_distribution(n_bits: int) -> Real:
    """``p(x) ∝ popcount(x) + 1`` ``[PTQNN Fig 5]``."""
    w = popcounts(n_bits) + 1.0
    return w / w.sum()


def target_activation(n_bits: int, scale: float = 1.0) -> Real:
    """Per-input target ``P(out=1 | x) = scale · (popcount+1)/(n_bits+1)`` for the iterative
    scheme: with uniform inputs any ``scale`` reproduces ``target_distribution`` after the
    renormalisation, but the sample-wise loss pins the absolute level, which the superposition
    loss does not — a strictly harder problem (``[ASSUMPTION]``; the paper's iterative baseline
    in Fig 6 is described only as "optimizing examples iteratively")."""
    return np.asarray(scale * (popcounts(n_bits) + 1.0) / (n_bits + 1.0), dtype=np.float64)


def network(n_bits: int = 5, n_hidden: int = 5, hidden: HiddenKind = "prototype") -> Network:
    return Network(n_inputs=n_bits, n_hidden=n_hidden, hidden=hidden)


def superposition_prep(net: Network) -> list[ir.Op]:
    """``H`` on every input — all ``2^n`` bit strings at once ``[PTQNN Fig 5]``."""
    return [ir.h(q) for q in net.input_qubits]


def basis_prep(net: Network, x: int) -> list[ir.Op]:
    """Prepare the computational-basis input ``x`` (qubit 0 = most significant bit)."""
    n = net.n_inputs
    return [ir.x(q) for q in range(n) if (x >> (n - 1 - q)) & 1]


def kl_loss(net: Network, backend: Backend | None = None) -> Loss:
    be = backend or ref.StatevectorBackend()
    target = target_distribution(net.n_inputs)
    prep = superposition_prep(net)

    def loss(params: Real) -> float:
        q = distribution.conditioned_input_distribution(net, params, prep, be)
        return losses.kl_divergence(target, q)

    return loss


def sample_loss(
    net: Network, x: int, backend: Backend | None = None, target_scale: float = 1.0
) -> Loss:
    """Squared error of ``P(out=1)`` against the target activation for one input ``x``."""
    be = backend or ref.StatevectorBackend()
    t = target_activation(net.n_inputs, target_scale)[x]
    prep = basis_prep(net, x)

    def loss(params: Real) -> float:
        p1 = distribution.output_activation(net, params, prep, be)
        return float((p1 - t) ** 2)

    return loss


# [ASSUMPTION] KW–SA hyper-parameters for bit-count; the paper publishes none.  Chosen so that
# the superposition run reaches KL < 1e-3 in O(10) iterations (Fig 6 shows ≈8).
# Grid-searched over a0 ∈ {20,50,100}, c0 ∈ {0.2,0.5}, (t0,cooling) ∈ {(0,·),(0.02,0.7),(0.05,0.8)}
# on 6 seeds: this setting had the lowest median KL after 12 sweeps (1.1e-3; worst seed 1.5e-2).
SUPERPOSITION_CONFIG = kwsa.KWSAConfig(
    iterations=20, a0=20.0, alpha=0.3, c0=0.2, gamma=0.101, t0=0.02, cooling=0.7, seed=0
)


def train_superposition(
    net: Network,
    seed: int = 0,
    iterations: int | None = None,
    backend: Backend | None = None,
    config: kwsa.KWSAConfig | None = None,
) -> kwsa.KWSAResult:
    cfg = config or SUPERPOSITION_CONFIG
    cfg = kwsa.KWSAConfig(
        **{**cfg.__dict__, "iterations": iterations or cfg.iterations, "seed": seed}
    )
    x0 = net.random_params(np.random.default_rng(seed))
    return kwsa.minimise(kl_loss(net, backend), x0, cfg)


def train_iterative(
    net: Network,
    seed: int = 0,
    iterations: int = 200,
    backend: Backend | None = None,
    config: kwsa.KWSAConfig | None = None,
    target_scale: float = 1.0,
) -> kwsa.KWSAResult:
    """Sample-wise baseline (Fig 6, red): each iteration takes one KW–SA step on the squared
    error of a single random input; the KL of the full distribution is recorded per iteration."""
    rng = np.random.default_rng(seed)
    cfg = config or SUPERPOSITION_CONFIG
    monitor = kl_loss(net, backend)
    x = net.random_params(rng)
    history = [monitor(x)]
    best_x, best_f, best_history = x.copy(), history[0], [history[0]]
    evaluations = 1
    for n in range(iterations):
        sample = int(rng.integers(2**net.n_inputs))
        step_cfg = kwsa.KWSAConfig(
            **{**cfg.__dict__, "iterations": 1, "seed": int(rng.integers(2**31))}
        )
        # one KW sweep on this sample at schedule position n (a_n, c_n decay as in the
        # superposition run); continue from the annealer's accepted state, not its best.
        res = kwsa.minimise(
            sample_loss(net, sample, backend, target_scale), x, step_cfg, start_iteration=n
        )
        x = res.final_params if res.final_params is not None else res.params
        evaluations += res.evaluations
        kl = monitor(x)
        history.append(kl)
        if kl < best_f:
            best_x, best_f = x.copy(), kl
        best_history.append(best_f)
    return kwsa.KWSAResult(
        params=best_x,
        loss=best_f,
        history=history,
        evaluations=evaluations,
        final_params=x,
        final_loss=history[-1],
        best_history=best_history,
    )
