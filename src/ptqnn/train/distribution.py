"""Read the paper's observables off a backend's outcome distribution.

``[PTQNN Fig 5, §III.C]``: "Hadamard gates are used to encode all possible input states in
superposition. The measured outcome distribution can be compared to the target distribution
using e.g. a KL-divergence score."  The histograms in Fig 5 are indexed by the *input* bit
string, so the measured distribution is the joint ``P(inputs = x, out = 1)`` renormalised over
``x`` (Fig 5 top-right sums to one over the 32 input strings).  Conditioning on ``out = 1`` is
what makes the output amplitude act as a weight on each input pattern — "an amplitude encoding
of the output" ``[PTQNN Fig 4 caption]``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ptqnn.backends import reference as ref
from ptqnn.backends.base import Backend
from ptqnn.circuits import ir
from ptqnn.circuits.neurons import Network

Real = npt.NDArray[np.float64]


def joint_input_output(
    net: Network, params: npt.ArrayLike, prep: Sequence[ir.Op], backend: Backend
) -> Real:
    """``P(inputs = x, out = o)`` as an array of shape ``(2**n_inputs, 2)``."""
    p = backend.probabilities(net.circuit(params, prep))
    m = ref.marginal(p, [*net.input_qubits, net.output_qubit])
    return m.reshape(2**net.n_inputs, 2)


def conditioned_input_distribution(
    net: Network, params: npt.ArrayLike, prep: Sequence[ir.Op], backend: Backend
) -> Real:
    """``P(x | out = 1)`` — the model distribution compared to the target in ``[PTQNN Eq 4]``."""
    joint = joint_input_output(net, params, prep, backend)[:, 1]
    z = joint.sum()
    if z <= 0:
        return np.full(joint.shape, 1.0 / joint.size)
    return joint / z


def output_activation(
    net: Network, params: npt.ArrayLike, prep: Sequence[ir.Op], backend: Backend
) -> float:
    """``P(out = 1)`` for a single prepared input — the sample-wise observable."""
    p = backend.probabilities(net.circuit(params, prep))
    return float(ref.marginal(p, [net.output_qubit])[1])
