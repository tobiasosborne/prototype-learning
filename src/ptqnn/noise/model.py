"""Noise models: attach CP maps to a circuit and wrap a backend.

A :class:`NoiseModel` turns an ideal :class:`~ptqnn.circuits.ir.Circuit` into a noisy one by
inserting :class:`~ptqnn.circuits.ir.Channel` ops:

* after every gate, on the gate's qubits: thermal relaxation for the gate duration (if ``T1``/
  ``T2`` are set), then the gate-class channel (``one_qubit`` / ``two_qubit`` / ``per_gate``) —
  the same composition order as Qiskit-Aer's ``thermal.compose(depolarizing)`` convention, so the
  two can be cross-checked (``tests/test_noise_model.py``);
* before a gate, on each of its qubits that has been waiting: an ``idle`` relaxation channel for
  the wait.  Gates are scheduled as-soon-as-possible with one clock per qubit, which is what a
  compiler would do with a list-scheduled circuit and what hardware does absent barriers;
* after the last gate all qubits idle until the latest clock (measurement is simultaneous).

Readout error is a classical confusion map on the outcome distribution and is applied by
:class:`NoisyBackend` after simulation.  ``[ASSUMPTION]``: vendors publish *average* gate
fidelities; we model the incoherent part as depolarising with matching average fidelity and the
time-dependent part as T1/T2 relaxation, which double counts slightly (the RB number already
contains some relaxation during the gate).  This is the standard practice (e.g. Aer's
``NoiseModel.from_backend``) and errs on the pessimistic side.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt

from ptqnn.backends.base import Backend
from ptqnn.circuits import ir
from ptqnn.noise import channels as ch

Kraus = ch.Kraus


@dataclass(frozen=True)
class Schedule:
    start_ns: list[float]
    total_ns: float


def _named(kraus: Kraus) -> str:
    # Best-effort label so noisy circuits stay readable in tests and transpiler output.
    n = len(kraus)
    d = kraus[0].shape[0]
    if n == d * d:
        return "depolarizing"
    return "pauli" if all(np.allclose(np.abs(k), np.abs(k).T) for k in kraus) else "channel"


@dataclass(frozen=True)
class NoiseModel:
    one_qubit: Kraus | None = None
    two_qubit: Kraus | None = None
    per_gate: Mapping[str, Kraus | None] = field(default_factory=dict)
    """Per gate-name override; ``None`` means "no channel for this gate"."""
    t1_us: float | None = None
    t2_us: float | None = None
    excited_population: float = 0.0
    duration_1q_ns: float = 0.0
    duration_2q_ns: float = 0.0
    durations_ns: Mapping[str, float] = field(default_factory=dict)
    """Per gate-name duration override (e.g. ``{"h": 35.0}``)."""
    readout: ch.ReadoutError | None = None
    idle_noise: bool = True
    name: str = "custom"

    def __post_init__(self) -> None:
        if (self.t1_us is None) != (self.t2_us is None):
            raise ValueError("T1 and T2 must be given together")

    # --- queries -------------------------------------------------------------------------------

    def is_trivial(self) -> bool:
        return (
            self.one_qubit is None
            and self.two_qubit is None
            and not any(v is not None for v in self.per_gate.values())
            and self.t1_us is None
            and (self.readout is None or self.readout.is_trivial())
        )

    @property
    def has_relaxation(self) -> bool:
        return self.t1_us is not None

    def duration_ns(self, gate: ir.Gate) -> float:
        if gate.name in self.durations_ns:
            return self.durations_ns[gate.name]
        return self.duration_1q_ns if gate.n_qubits == 1 else self.duration_2q_ns

    def gate_channel(self, gate: ir.Gate) -> Kraus | None:
        if gate.name in self.per_gate:
            return self.per_gate[gate.name]
        if gate.n_qubits == 1:
            return self.one_qubit
        if gate.n_qubits == 2:
            return self.two_qubit
        return None

    def relaxation(self, duration_ns: float) -> Kraus | None:
        if self.t1_us is None or self.t2_us is None or duration_ns <= 0:
            return None
        return ch.thermal_relaxation(
            self.t1_us, self.t2_us, duration_ns * 1e-3, self.excited_population
        )

    # --- transformation ------------------------------------------------------------------------

    def schedule(self, circuit: ir.Circuit) -> Schedule:
        """ASAP start time of every op (ns) and the total duration."""
        clock = [0.0] * circuit.n_qubits
        starts: list[float] = []
        for op in circuit.ops:
            s = max(clock[q] for q in op.qubits)
            starts.append(s)
            d = self.duration_ns(op) if isinstance(op, ir.Gate) else 0.0
            for q in op.qubits:
                clock[q] = s + d
        return Schedule(starts, max(clock) if clock else 0.0)

    def apply(self, circuit: ir.Circuit) -> ir.Circuit:
        """The noisy circuit: ideal ops interleaved with the model's channels."""
        out = ir.Circuit(circuit.n_qubits, initial_state=circuit.initial_state)
        clock = [0.0] * circuit.n_qubits
        for op in circuit.ops:
            if not isinstance(op, ir.Gate):
                out.append(op)  # pre-existing channels pass through untouched
                continue
            start = max(clock[q] for q in op.qubits)
            if self.idle_noise and self.has_relaxation:
                for q in op.qubits:
                    wait = start - clock[q]
                    idle = self.relaxation(wait)
                    if idle is not None:
                        out.append(ir.Channel((q,), idle, "idle"))
            out.append(op)
            d = self.duration_ns(op)
            for q in op.qubits:
                clock[q] = start + d
            relax = self.relaxation(d)
            if relax is not None:
                for q in op.qubits:
                    out.append(ir.Channel((q,), relax, "relax"))
            chan = self.gate_channel(op)
            if chan is not None:
                out.append(ir.Channel(op.qubits, chan, _named(chan)))
        if self.idle_noise and self.has_relaxation:
            end = max(clock)
            for q in range(circuit.n_qubits):
                idle = self.relaxation(end - clock[q])
                if idle is not None:
                    out.append(ir.Channel((q,), idle, "idle"))
        return out

    def scaled(self, factor: float) -> NoiseModel:
        """Noise strength multiplied by ``factor`` (depolarising ``p``, ``1/T1``, ``1/T2``,
        readout error); used for noise sweeps and zero-noise-extrapolation style studies."""

        def scale_kraus(k: Kraus | None) -> Kraus | None:
            if k is None:
                return None
            d = k[0].shape[0]
            p = 1 - ch.average_gate_fidelity(k)
            n = round(np.log2(d))
            return ch.depolarizing(min(1.0, ch.depolarizing_probability(1 - factor * p, n)), n)

        return replace(
            self,
            one_qubit=scale_kraus(self.one_qubit),
            two_qubit=scale_kraus(self.two_qubit),
            per_gate={k: scale_kraus(v) for k, v in self.per_gate.items()},
            t1_us=None if self.t1_us is None else self.t1_us / factor,
            t2_us=None if self.t2_us is None else self.t2_us / factor,
            readout=None
            if self.readout is None
            else ch.ReadoutError(
                _scale(self.readout.p01, factor), _scale(self.readout.p10, factor)
            ),
            name=f"{self.name}×{factor:g}",
        )


def _scale(v: float | tuple[float, ...], f: float) -> float | tuple[float, ...]:
    if isinstance(v, tuple):
        return tuple(min(1.0, x * f) for x in v)
    return min(1.0, v * f)


class NoisyBackend:
    """``Backend`` that runs ``model.apply(circuit)`` on ``inner`` and applies readout error."""

    def __init__(self, inner: Backend, model: NoiseModel):
        self.inner = inner
        self.model = model

    @property
    def name(self) -> str:
        return f"noisy({self.inner.name}, {self.model.name})"

    def probabilities(self, circuit: ir.Circuit) -> npt.NDArray[np.float64]:
        p = self.inner.probabilities(self.model.apply(circuit))
        if self.model.readout is not None and not self.model.readout.is_trivial():
            p = self.model.readout.apply(p, circuit.n_qubits)
        return p
