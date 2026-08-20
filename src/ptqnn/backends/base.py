"""Backend protocol: anything that turns a :class:`~ptqnn.circuits.ir.Circuit` into outcome
probabilities.  Training code depends only on this, so the same experiment runs noiselessly
(statevector), under a CP-map noise model (density matrix / orkan / trajectories) or, later,
against hardware through an executor that returns counts.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from ptqnn.circuits.ir import Circuit


@runtime_checkable
class Backend(Protocol):
    """Returns the computational-basis distribution (qubit 0 = most significant bit)."""

    @property
    def name(self) -> str: ...

    def probabilities(self, circuit: Circuit) -> npt.NDArray[np.float64]: ...
