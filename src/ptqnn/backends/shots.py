"""A sampling backend: finite-shot estimates of any inner backend's distribution, with
*common random numbers* for finite-difference probe pairs.

On hardware ``P(·)`` is never known — only ``counts/N``.  Wrapping a backend in
:class:`ShotBackend` makes every training loop in the repo run on such estimates unchanged.

Common random numbers (shot-noise programme, item 2).  Kiefer–Wolfowitz estimates a derivative
from ``(f(x + c) − f(x − c)) / 2c``.  With independent shot noise of variance ``σ²`` on each
evaluation the estimate has variance ``σ²/(2c²)`` — at ``c = 0.2`` that is 12.5 σ², which is why
finite-difference optimisation on a sampled backend stalls.  If both probes are evaluated with the
*same* random draws (the same ``u_i`` fed to the inverse CDF of the two nearby distributions),
the noise largely cancels and the variance becomes ``O(c²/N)`` (the two multinomials are coupled
through the same uniforms, so ``f(x+c) − f(x−c)`` only sees the draws that fall between the two
CDFs).  On hardware the analogue is interleaving the ``±c`` circuits in one job so they share
drift; on a simulator it is :meth:`ShotBackend.common_random_numbers`, which
:func:`ptqnn.train.kwsa.minimise` enters for every coordinate's probe triple via
``probe_context``.

Inside a :meth:`common_random_numbers` block the *first* call is sampled by inverse CDF from a
fixed array of uniforms and every later call is drawn from the **maximal coupling** with that
first sample: shot ``j`` keeps its reference outcome ``i`` with probability
``min(1, p_new[i]/p_ref[i])`` and is otherwise redrawn from the residual ``(p_new − p_ref)₊``, so
the expected number of shots that change outcome is ``N · TV(p_new, p_ref)`` — the smallest any
coupling can achieve (Lindvall, *Lectures on the Coupling Method*, 1992).  Inverse-CDF sampling
with shared uniforms alone is *not* enough here: a parameter change moves mass between distant
basis indices, so the shared-uniform coupling changes ``N · Σ_i |F_new(i) − F_ref(i)|`` draws,
which for the bit-count loss was measured at only a 1.3× variance reduction; the maximal
coupling is what delivers the ``O(c²)`` behaviour.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import numpy.typing as npt

from ptqnn.backends.base import Backend
from ptqnn.circuits.ir import Circuit

Real = npt.NDArray[np.float64]


def outcomes_from_uniforms(p: Real, u: Real) -> npt.NDArray[np.int64]:
    """Inverse-CDF sampling: the outcome of uniform ``u_j`` is the first bin whose cumulative
    probability exceeds ``u_j``."""
    q = np.clip(np.asarray(p, dtype=np.float64), 0.0, None)
    cdf = np.cumsum(q / q.sum())
    idx = np.searchsorted(cdf, u, side="right")
    return np.minimum(idx, q.size - 1).astype(np.int64)


def counts_from_uniforms(p: Real, u: Real) -> npt.NDArray[np.int64]:
    """Histogram of :func:`outcomes_from_uniforms`."""
    return np.bincount(outcomes_from_uniforms(p, u), minlength=np.asarray(p).size).astype(np.int64)


def coupled_counts(
    p_new: Real, p_ref: Real, out_ref: npt.NDArray[np.int64], w: Real, r: Real
) -> npt.NDArray[np.int64]:
    """Maximal coupling of a ``p_new`` sample to the reference outcomes ``out_ref ~ p_ref``.

    Shot ``j`` keeps ``out_ref[j] = i`` iff ``w_j < min(1, p_new[i]/p_ref[i])``; otherwise it is
    redrawn (uniform ``r_j``, inverse CDF) from the residual ``(p_new − p_ref)₊ / TV``.  The
    marginal of the result is exactly ``p_new`` and ``E[#changed] = N · TV(p_new, p_ref)``.
    """
    keep_prob = np.ones_like(p_new)
    nz = p_ref > 0
    keep_prob[nz] = np.minimum(1.0, p_new[nz] / p_ref[nz])
    keep = w < keep_prob[out_ref]
    out = out_ref.copy()
    n_move = int((~keep).sum())
    if n_move:
        resid = np.clip(p_new - p_ref, 0.0, None)
        if resid.sum() <= 0:  # identical distributions up to rounding: nothing to move to
            return np.bincount(out, minlength=p_new.size).astype(np.int64)
        out[~keep] = outcomes_from_uniforms(resid, r[~keep])
    return np.bincount(out, minlength=p_new.size).astype(np.int64)


class ShotBackend:
    """``probabilities(circuit) = counts / shots`` with ``counts`` drawn from ``inner``'s exact
    distribution.  Satisfies :class:`~ptqnn.backends.base.Backend`."""

    def __init__(self, inner: Backend, shots: int, seed: int = 0) -> None:
        if shots <= 0:
            raise ValueError("shots must be positive")
        self.inner = inner
        self.shots = int(shots)
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)
        self._in_block = False
        self._ref: tuple[Real, npt.NDArray[np.int64], Real, Real] | None = None
        self.shots_used = 0
        self.calls = 0

    @property
    def name(self) -> str:
        return f"{self.inner.name}+shots={self.shots}"

    def probabilities(self, circuit: Circuit) -> Real:
        p = np.clip(self.inner.probabilities(circuit), 0.0, None)
        p = p / p.sum()
        self.shots_used += self.shots
        self.calls += 1
        if not self._in_block:
            return counts_from_uniforms(p, self._rng.random(self.shots)) / self.shots
        if self._ref is None:
            u = self._rng.random(self.shots)
            outcomes = outcomes_from_uniforms(p, u)
            self._ref = (p, outcomes, self._rng.random(self.shots), self._rng.random(self.shots))
            return np.bincount(outcomes, minlength=p.size) / self.shots
        p_ref, out_ref, w, r = self._ref
        return coupled_counts(p, p_ref, out_ref, w, r) / self.shots

    @contextmanager
    def common_random_numbers(self) -> Iterator[None]:
        """Inside this block the first call fixes the reference sample and every later call is
        maximally coupled to it (see the module docstring); each block starts afresh."""
        if self._in_block:
            raise RuntimeError("common_random_numbers blocks do not nest")
        self._in_block = True
        self._ref = None
        try:
            yield
        finally:
            self._in_block = False
            self._ref = None
