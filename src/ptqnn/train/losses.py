"""Distribution dissimilarities used as training losses.

``[PTQNN Eq 4]``: ``D_KL(P‖Q) = Σ_x p(x) log (p(x)/q(x))`` — "In our experiments we use the
standard KL-divergence as optimization criteria to perform a global optimization of the network
weights."  ``[PTQNN Eq 5]``: ``D_cos(P, Q) = 1 − P·Q`` for unit vectors, "a useful alternative
for quantum computers, as this measure is also directly expressible as quantum gate sequence".
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Real = npt.NDArray[np.float64]


def kl_divergence(target: Real, model: Real, eps: float = 1e-12) -> float:
    """``Σ p log(p/q)`` with ``0 log 0 = 0``; ``q`` is floored at ``eps`` so the value stays finite
    when the model assigns zero mass to a target outcome (``[ASSUMPTION]``: the paper does not
    say how it regularises)."""
    p = np.asarray(target, dtype=np.float64)
    q = np.maximum(np.asarray(model, dtype=np.float64), eps)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def cosine_dissimilarity(p: Real, q: Real) -> float:
    """``1 − P·Q`` after normalising both to unit Euclidean norm."""
    a = np.asarray(p, dtype=np.float64)
    b = np.asarray(q, dtype=np.float64)
    return float(1.0 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def squared_error(p: Real, q: Real) -> float:
    """Plain ``Σ (p−q)²`` — the per-sample ℓ2 loss of the iterative scheme ``[SNN §III.A]``."""
    a = np.asarray(p, dtype=np.float64)
    b = np.asarray(q, dtype=np.float64)
    return float(np.sum((a - b) ** 2))
