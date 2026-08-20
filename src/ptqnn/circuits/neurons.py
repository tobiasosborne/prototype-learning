"""Stochastic neurons and the shallow network of the paper.

Classic (stochastic) perceptron ``[PTQNN Fig 1, §II.B]``: "Two input qubits are connected to an
output qubit by using controlled rotation (CRX) gates. A plain rotation realizes the bias."
Ops: ``RX(bias)`` on the output qubit, then ``CRX(w_i)`` from each input.  For a basis input
``b`` all rotations on the output commute, so ``P(out=1) = sin²((bias + Σ_i b_i w_i)/2)``: only
1-valued inputs contribute — the asymmetry the paper sets out to fix.

Prototype perceptron ``[PTQNN Fig 2, §III.B]``: per input ``i`` — ``RX(φ_i)`` on the input,
``CRX(θ)`` input→output, ``RX(−φ_i)`` on the input. "Rotations before the CR-rotations modify
the likelihood for increasing/decreasing an activation on |q3⟩. This allows to capture also
0-entries as valuable information."  The same ``θ`` is used for every CRX of one neuron, and
there is no separate bias: "the former bias is shifted to the amount of activation when a
rotation is triggered and the parameters of the controlled rotations are shifted to the
rotations before and after the CR-gate."  Parameter count per neuron: ``n_inputs + 1`` — the
same as a classic neuron (``n_inputs`` weights + bias), cf. "with the same amount of
parameters" ``[PTQNN §III.C]``.

Network ``[PTQNN Fig 4, 5, 10, 12]``: inputs → ``n_hidden`` neurons, each reading *all* inputs
→ one output neuron that is always a classic perceptron over the hidden qubits (the figures show
a bias ``RX`` on the last wire followed by one ``CRX`` per hidden qubit).  Qubit layout: inputs
``0..n_in-1``, hidden ``n_in..n_in+n_hid-1``, output last.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from ptqnn.circuits import ir

HiddenKind = Literal["prototype", "classic"]


def classic_neuron(
    inputs: Sequence[int], out: int, weights: Sequence[float] | npt.NDArray[np.float64], bias: float
) -> list[ir.Gate]:
    """``[PTQNN Fig 1]``: ``RX(bias)`` on ``out``; ``CRX(w_i)`` ``inputs[i] → out``."""
    if len(weights) != len(inputs):
        raise ValueError("one weight per input")
    ops = [ir.rx(out, bias)]
    ops += [ir.crx(i, out, w) for i, w in zip(inputs, weights, strict=True)]
    return ops


def prototype_neuron(
    inputs: Sequence[int],
    out: int,
    prototype: Sequence[float] | npt.NDArray[np.float64],
    theta: float,
) -> list[ir.Gate]:
    """``[PTQNN Fig 2]``: for each input ``RX(φ_i)``, ``CRX(θ) → out``, ``RX(−φ_i)``."""
    if len(prototype) != len(inputs):
        raise ValueError("one prototype angle per input")
    ops: list[ir.Gate] = []
    for i, phi in zip(inputs, prototype, strict=True):
        ops += [ir.rx(i, phi), ir.crx(i, out, theta), ir.rx(i, -phi)]
    return ops


@dataclass(frozen=True)
class Network:
    """Shallow network: ``n_inputs`` → ``n_hidden`` (prototype or classic) → 1 classic output.

    The flat parameter vector is laid out hidden-neuron by hidden-neuron, each as
    ``(n_inputs per-input angles..., θ-or-bias)``, followed by the output neuron as
    ``(n_hidden weights..., bias)``.  ``pack``/``unpack`` convert.
    """

    n_inputs: int
    n_hidden: int
    hidden: HiddenKind = "prototype"

    @property
    def n_qubits(self) -> int:
        return self.n_inputs + self.n_hidden + 1

    @property
    def input_qubits(self) -> tuple[int, ...]:
        return tuple(range(self.n_inputs))

    @property
    def hidden_qubits(self) -> tuple[int, ...]:
        return tuple(range(self.n_inputs, self.n_inputs + self.n_hidden))

    @property
    def output_qubit(self) -> int:
        return self.n_inputs + self.n_hidden

    @property
    def n_params(self) -> int:
        return self.n_hidden * (self.n_inputs + 1) + (self.n_hidden + 1)

    def pack(
        self,
        hidden: Sequence[tuple[Sequence[float], float]],
        output: tuple[Sequence[float], float],
    ) -> npt.NDArray[np.float64]:
        flat: list[float] = []
        for angles, last in hidden:
            flat += [*angles, last]
        flat += [*output[0], output[1]]
        arr = np.asarray(flat, dtype=np.float64)
        if arr.size != self.n_params:
            raise ValueError(f"params: expected {self.n_params}, got {arr.size}")
        return arr

    def unpack(
        self, params: npt.ArrayLike
    ) -> tuple[list[tuple[npt.NDArray[np.float64], float]], tuple[npt.NDArray[np.float64], float]]:
        p = np.asarray(params, dtype=np.float64).ravel()
        if p.size != self.n_params:
            raise ValueError(f"params: expected {self.n_params}, got {p.size}")
        k = self.n_inputs + 1
        hidden = [
            (p[j * k : j * k + self.n_inputs], float(p[j * k + self.n_inputs]))
            for j in range(self.n_hidden)
        ]
        tail = p[self.n_hidden * k :]
        return hidden, (tail[: self.n_hidden], float(tail[self.n_hidden]))

    def circuit(
        self,
        params: npt.ArrayLike,
        prep: Sequence[ir.Op] = (),
        initial_state: npt.NDArray[np.complex128] | None = None,
    ) -> ir.Circuit:
        """Full circuit: optional input preparation ``prep`` (gates) or ``initial_state``
        (a statevector on all ``n_qubits``), hidden layer, output neuron."""
        hidden, (w_out, b_out) = self.unpack(params)
        circ = ir.Circuit(self.n_qubits, initial_state=initial_state)
        circ.extend(prep)
        for hq, (angles, last) in zip(self.hidden_qubits, hidden, strict=True):
            if self.hidden == "prototype":
                circ.extend(prototype_neuron(self.input_qubits, hq, angles, last))
            else:
                circ.extend(classic_neuron(self.input_qubits, hq, angles, last))
        circ.extend(classic_neuron(self.hidden_qubits, self.output_qubit, w_out, b_out))
        return circ

    def random_params(
        self, rng: np.random.Generator, scale: float = np.pi
    ) -> npt.NDArray[np.float64]:
        return rng.uniform(-scale, scale, size=self.n_params)
