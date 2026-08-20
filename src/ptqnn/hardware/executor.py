"""Execution adapters: one protocol, several back ends, no credentials assumed.

An :class:`Executor` turns a circuit into **counts** — ``{bitstring: shots}`` with the bit string
in :data:`ptqnn.hardware.qasm.BITSTRING_ORDER` (qubit 0 leftmost, matching ``[PTQNN Fig 5]``).
That is the only thing hardware ever returns, so it is the only thing the protocol promises.

* :class:`LocalExecutor` — our own backends plus ``reference.sample_counts``; the oracle.
* :class:`QiskitAerExecutor` — an independent implementation of the same distribution, used as a
  cross-check (``tests/test_hardware.py`` compares it to ``LocalExecutor`` in total-variation
  distance at 20 000 shots).  Our qubit ``q`` becomes Qiskit qubit ``n-1-q``, which makes
  Qiskit's little-endian significance agree with our big-endian one, so Aer's count keys need
  **no** re-ordering (unlike the OpenQASM route — see :mod:`ptqnn.hardware.qasm`).
* :class:`IBMRuntimeExecutor` — dry run only.  ``qiskit_ibm_runtime`` is not installed and no
  token is configured, so the executor builds the ISA circuit with ``qiskit.transpile`` against
  an explicit basis + coupling map and returns the payload it *would* submit.  It never opens a
  socket.
* :class:`BraketExecutor`, :class:`QuantinuumExecutor` — stubs that say exactly which package and
  which credential is missing.

:func:`cost_estimate` prices a compiled circuit.  Every vendor formula carries its URL; anything
we could not source from vendor documentation is marked ``[ASSUMPTION]`` in the returned dict and
in ``PLAN.md §0``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from ptqnn.backends import reference
from ptqnn.backends.base import Backend
from ptqnn.circuits import ir
from ptqnn.hardware import qasm
from ptqnn.hardware.transpile import CompiledCircuit, heavy_hex_patch, metrics

if TYPE_CHECKING:  # pragma: no cover - typing only
    from qiskit import QuantumCircuit

Counts = dict[str, int]


@runtime_checkable
class Executor(Protocol):
    """Anything that can be handed a circuit and a shot budget."""

    def run(self, circuit: ir.Circuit, shots: int) -> Counts: ...

    def describe(self) -> dict[str, Any]: ...


def counts_to_probabilities(counts: Mapping[str, int], n_qubits: int) -> np.ndarray:
    """Counts (our bit-string order) → a length-``2^n`` probability vector in IR index order."""
    p = np.zeros(2**n_qubits, dtype=np.float64)
    total = sum(counts.values())
    if total == 0:
        return p
    for key, c in counts.items():
        p[int(key, 2)] = c / total
    return p


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """``½‖p-q‖₁`` — the metric the executor cross-check is stated in."""
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


# --- local ------------------------------------------------------------------------------------


@dataclass
class LocalExecutor:
    """Sample from one of our own backends.  Noise, if any, lives inside ``backend``."""

    backend: Backend
    seed: int | None = None

    def run(self, circuit: ir.Circuit, shots: int) -> Counts:
        rng = np.random.default_rng(self.seed)
        p = self.backend.probabilities(circuit)
        c = reference.sample_counts(p, shots, rng)
        n = circuit.n_qubits
        return {format(i, f"0{n}b"): int(v) for i, v in enumerate(c) if v}

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "local",
            "backend": self.backend.name,
            "seed": self.seed,
            "bitstring_order": qasm.BITSTRING_ORDER,
        }


# --- qiskit bridge ----------------------------------------------------------------------------

_QISKIT_1Q = {"rx", "ry", "rz", "h", "x", "y", "z", "sx", "s", "sdg", "t", "tdg", "id"}
_QISKIT_2Q = {"cx", "cz", "swap", "crx", "cry", "crz", "iswap"}


def to_qiskit(circuit: ir.Circuit) -> QuantumCircuit:
    """Our IR → a Qiskit circuit with **reversed qubit order** (our ``q`` → Qiskit ``n-1-q``).

    The reversal is what makes the two significance conventions agree: Qiskit's qubit 0 is the
    least significant bit, ours is the most significant, so sending our qubit 0 to Qiskit's
    qubit ``n-1`` puts it back at the top of the basis index.  Consequently
    ``Operator(to_qiskit(c)).data`` equals the IR unitary directly, and Aer's count keys are
    already in :data:`ptqnn.hardware.qasm.BITSTRING_ORDER`.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate

    n = circuit.n_qubits
    qc = QuantumCircuit(n)

    def w(q: int) -> int:
        return n - 1 - q

    for op in circuit.ops:
        if not isinstance(op, ir.Gate):
            raise ValueError(f"cannot convert channel {op.name!r} to a Qiskit circuit")
        name, qs, ps = op.name, [w(q) for q in op.qubits], op.params
        if name in _QISKIT_1Q or name in _QISKIT_2Q:
            getattr(qc, name)(*ps, *qs)
        else:
            # Arbitrary gate: our matrix orders qubits most-significant first, Qiskit's
            # ``unitary`` takes them least-significant first, so the wire list is reversed.
            qc.append(UnitaryGate(op.matrix, label=name), qs[::-1])
    return qc


@dataclass
class QiskitAerExecutor:
    """Sample with qiskit-aer, optionally under an Aer ``NoiseModel``.

    ``noise_model`` is an ``qiskit_aer.noise.NoiseModel`` (typed ``Any`` so importing this module
    never requires qiskit).  Independent of our simulator all the way down — that is the point.
    """

    noise_model: Any = None
    seed: int | None = None
    method: str = "automatic"

    def run(self, circuit: ir.Circuit, shots: int) -> Counts:
        from qiskit import transpile as qiskit_transpile
        from qiskit_aer import AerSimulator

        qc = to_qiskit(circuit)
        qc.measure_all()
        sim = AerSimulator(noise_model=self.noise_model, method=self.method)
        isa = qiskit_transpile(qc, sim, seed_transpiler=self.seed)
        result = sim.run(isa, shots=shots, seed_simulator=self.seed).result()
        # measure_all() maps Qiskit qubit j → c[j], and Aer renders c[n-1] leftmost; because
        # to_qiskit() already reversed the wires, that leftmost bit is our qubit 0.
        return {k.replace(" ", ""): int(v) for k, v in result.get_counts().items()}

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "qiskit-aer",
            "method": self.method,
            "noisy": self.noise_model is not None,
            "seed": self.seed,
            "bitstring_order": qasm.BITSTRING_ORDER,
        }


# --- IBM Runtime (dry run) --------------------------------------------------------------------

IBM_HERON_BASIS = ("rz", "sx", "x", "cz", "id")


@dataclass
class IBMRuntimeExecutor:
    """Build the ISA circuit an IBM Quantum job would carry — without a network call.

    ``qiskit_ibm_runtime`` is not installed in this project, so there is no ``FakeHeron`` to
    borrow a coupling map from; we transpile against an explicit basis (:data:`IBM_HERON_BASIS`,
    sourced in :mod:`ptqnn.hardware.gatesets`) and a heavy-hex patch of the right size.  With
    ``dry_run=True`` (the only supported mode) :meth:`run` refuses and :meth:`dry_run_payload`
    returns the transpiled metrics plus the exact payload fields a ``SamplerV2`` job would take.
    """

    backend_name: str = "ibm_heron"
    dry_run: bool = True
    coupling: Sequence[tuple[int, int]] | None = None
    basis_gates: Sequence[str] = IBM_HERON_BASIS
    optimization_level: int = 3
    seed: int | None = 0

    def _isa(self, circuit: ir.Circuit) -> tuple[QuantumCircuit, list[tuple[int, int]]]:
        from qiskit import transpile as qiskit_transpile
        from qiskit.transpiler import CouplingMap

        coupling = (
            list(self.coupling) if self.coupling is not None else heavy_hex_patch(circuit.n_qubits)
        )
        cmap = CouplingMap([[a, b] for a, b in coupling] + [[b, a] for a, b in coupling])
        isa = qiskit_transpile(
            to_qiskit(circuit),
            basis_gates=list(self.basis_gates),
            coupling_map=cmap,
            optimization_level=self.optimization_level,
            seed_transpiler=self.seed,
        )
        return isa, coupling

    def dry_run_payload(self, circuit: ir.Circuit, shots: int) -> dict[str, Any]:
        """The job body, the ISA circuit's OpenQASM 3, and its metrics.  Contacts nothing."""
        from qiskit import qasm3 as qiskit_qasm3

        isa, coupling = self._isa(circuit)
        ops = dict(isa.count_ops())
        two_q = sum(v for k, v in ops.items() if k in ("cz", "ecr", "cx"))
        return {
            "would_submit": {
                "primitive": "SamplerV2",
                "backend": self.backend_name,
                "pubs": [["<isa_circuit>", None, shots]],
                "shots": shots,
                "options": {"default_shots": shots, "dynamical_decoupling": {"enable": False}},
            },
            "isa": {
                "n_qubits": isa.num_qubits,
                "depth": isa.depth(),
                "ops": ops,
                "n_2q": two_q,
                "basis_gates": list(self.basis_gates),
                "coupling_map": [list(e) for e in coupling],
                "qasm3": qiskit_qasm3.dumps(isa),
            },
            "network": "none — dry run; qiskit_ibm_runtime is not installed and no token is set",
        }

    def run(self, circuit: ir.Circuit, shots: int) -> Counts:
        raise NotImplementedError(
            "IBMRuntimeExecutor is dry-run only.  A real submission needs: "
            "(1) `pip install qiskit-ibm-runtime`; (2) an IBM Quantum Platform API token in "
            "`QiskitRuntimeService(channel='ibm_quantum_platform', token=...)` or "
            "~/.qiskit/qiskit-ibm.json; (3) an instance CRN with QPU time on the plan; "
            "(4) a real backend so the ISA circuit is transpiled against the *live* coupling map "
            "and calibration, not our heavy-hex stand-in.  Use dry_run_payload() meanwhile."
        )

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "ibm-runtime",
            "backend": self.backend_name,
            "dry_run": self.dry_run,
            "basis_gates": list(self.basis_gates),
            "installed": False,
            "bitstring_order": qasm.BITSTRING_ORDER,
        }


# --- stubs ------------------------------------------------------------------------------------


@dataclass
class BraketExecutor:
    """AWS Braket (IonQ / IQM / Rigetti).  Not implemented — see the error message."""

    device_arn: str = "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1"
    s3_bucket: str = ""
    dry_run: bool = True

    _NEEDS = (
        "pip install amazon-braket-sdk",
        "AWS credentials with braket:CreateQuantumTask (env AWS_ACCESS_KEY_ID / "
        "AWS_SECRET_ACCESS_KEY or an instance role)",
        "an S3 bucket named `amazon-braket-*` in the device's region for result objects",
        "the device ARN and its availability window (QPUs are not always online)",
        "Braket bills per task plus per shot even for a failed task — see cost_estimate()",
    )

    def run(self, circuit: ir.Circuit, shots: int) -> Counts:
        raise NotImplementedError(
            "BraketExecutor is a stub.  To submit for real you need:\n  - "
            + "\n  - ".join(self._NEEDS)
            + "\nCircuits export via ptqnn.hardware.qasm.to_openqasm3 (Braket accepts OpenQASM 3)."
        )

    def describe(self) -> dict[str, Any]:
        return {"kind": "braket", "device_arn": self.device_arn, "requires": list(self._NEEDS)}


@dataclass
class QuantinuumExecutor:
    """Quantinuum Nexus / H-series.  Not implemented — see the error message."""

    machine: str = "H1-1E"
    dry_run: bool = True

    _NEEDS = (
        "pip install pytket pytket-quantinuum (or qnexus)",
        "a Quantinuum user account and password for `QuantinuumBackend.login()`; "
        "federated/Microsoft-ID accounts additionally need an API key",
        "an HQC allocation on the machine (the emulators H1-1E/H2-1E also consume HQCs)",
        "the circuit rebased to {U1q, Rz, ZZ/Rzz} — ptqnn.hardware.transpile.to_gateset does "
        "this; pytket's `QuantinuumBackend.get_compiled_circuit` would redo it",
    )

    def run(self, circuit: ir.Circuit, shots: int) -> Counts:
        raise NotImplementedError(
            f"QuantinuumExecutor({self.machine!r}) is a stub.  To submit for real you need:\n  - "
            + "\n  - ".join(self._NEEDS)
        )

    def describe(self) -> dict[str, Any]:
        return {"kind": "quantinuum", "machine": self.machine, "requires": list(self._NEEDS)}


# --- cost -------------------------------------------------------------------------------------

_SOURCES = {
    "quantinuum": "https://docs.quantinuum.com/systems/user_guide/hardware_user_guide/workflow.html",
    "ionq_azure": "https://learn.microsoft.com/en-us/azure/quantum/pricing",
    "braket": "https://aws.amazon.com/braket/pricing/",
}

# AWS Braket per-shot prices, USD, from the pricing page (retrieved 2026-08-20).  Every QPU task
# also costs a flat $0.30.  IonQ Aria and Rigetti Ankaa-3 have no current price row and are
# deliberately absent rather than guessed.
_BRAKET_PER_SHOT: dict[str, float] = {
    "ionq_forte": 0.08,
    "iqm_garnet": 0.00145,
    "iqm_emerald": 0.00160,
    "rigetti_cepheus": 0.000425,
}
_BRAKET_PER_TASK = 0.30


@dataclass(frozen=True)
class CostEstimate:
    model: str
    unit: str
    value: float
    breakdown: dict[str, float]
    sources: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "unit": self.unit,
            "value": self.value,
            "breakdown": dict(self.breakdown),
            "sources": list(self.sources),
            "assumptions": list(self.assumptions),
        }


def cost_estimate(
    compiled: CompiledCircuit,
    shots: int,
    device: object,
    *,
    error_mitigation: bool = False,
    job_overhead_s: float = 30.0,
) -> CostEstimate:
    """Price ``shots`` executions of ``compiled`` on ``device``.

    Vendor-specific where a formula is published, and a plain
    ``shots × circuit_time + per-job overhead`` wall-clock model otherwise.  The wall-clock model
    is ``[ASSUMPTION]``: real queues, calibration windows and (for QCCD machines) ion transport
    dominate, and no vendor publishes an end-to-end throughput figure.
    """
    m = compiled.metrics
    n1, n2 = int(m["n_1q"] or 0), int(m["n_2q"] or 0)
    nq = int(m["n_qubits"] or 0)
    vendor = str(getattr(device, "vendor", "")).lower()
    name = str(getattr(device, "name", compiled.device_name)).lower()

    # Finding 10: key the Braket table on the device slug alone.  No shipped device TOML has
    # "aws" or "braket" in its ``vendor`` field (they say IonQ, IQM, Rigetti), so gating on the
    # vendor string made every entry here unreachable.
    slug = name.replace("-", "_")
    if slug in _BRAKET_PER_SHOT:
        usd = _BRAKET_PER_TASK + _BRAKET_PER_SHOT[slug] * shots
        return CostEstimate(
            model="aws_braket",
            unit="USD",
            value=usd,
            breakdown={
                "per_task": _BRAKET_PER_TASK,
                "per_shot": _BRAKET_PER_SHOT[slug],
                "shots": shots,
            },
            sources=(_SOURCES["braket"],),
        )

    if "quantinuum" in vendor or name.startswith("quantinuum"):
        # "HQC = 5 + (N_1q + 10 N_2q + 5 N_m)/5000 * C" with N_m the state-prep/measurement ops.
        n_m = 2 * nq  # implicit initial preparation + the final measurement of every qubit
        hqc = 5.0 + (n1 + 10 * n2 + 5 * n_m) / 5000.0 * shots
        return CostEstimate(
            model="quantinuum_hqc",
            unit="HQC",
            value=hqc,
            breakdown={"base": 5.0, "n_1q": n1, "n_2q": n2, "n_m": n_m, "shots": shots},
            sources=(_SOURCES["quantinuum"],),
            assumptions=(
                "[ASSUMPTION] N_m = 2·n_qubits: the docs count 'state preparation and "
                "measurement operations, including the implicit initial state preparation'; we "
                "read that as one prep plus one measurement per qubit.",
            ),
        )

    if "ionq" in vendor or name.startswith("ionq"):
        # Azure Quantum's AQT model for IonQ; IonQ's own site publishes no formula.
        if "forte" in name:
            c1, c2, minimum = 0.0001645, 0.001121, (168.195 if error_mitigation else 25.7899)
            sub = "forte"
        else:
            c1, c2, minimum = 0.000220, 0.000975, (97.50 if error_mitigation else 12.4166)
            sub = "aria"
        gates = c1 * n1 * shots + c2 * n2 * shots
        return CostEstimate(
            model=f"ionq_azure_aqt_{sub}",
            unit="USD",
            value=minimum + gates,
            breakdown={"minimum": minimum, "gate_cost": gates, "n_1q": n1, "n_2q": n2},
            sources=(_SOURCES["ionq_azure"],),
            assumptions=(
                "[ASSUMPTION] the published formula reads 'AQT = m + c1·(N_1q·C) + c2·(N_2q·C)'; "
                "we treat m as an additive minimum charge rather than a floor.",
                "[ASSUMPTION] Azure's IonQ pricing is used as a proxy for IonQ direct/Braket.",
            ),
        )

    if "ibm" in vendor or name.startswith("ibm"):
        rate = 1.60
        seconds = _wall_clock_s(m, shots, job_overhead_s)
        return CostEstimate(
            model="ibm_pay_as_you_go",
            unit="USD",
            value=rate * seconds,
            breakdown={"usd_per_second": rate, "seconds": seconds},
            sources=(),
            assumptions=(
                "[ASSUMPTION] USD 1.60 per runtime second is IBM's 2022 launch figure; IBM's "
                "current pricing pages document per-second billing but publish no rate, so this "
                "number is UNVERIFIED and the estimate is order-of-magnitude only.",
                "[ASSUMPTION] runtime seconds = shots × critical-path time + job overhead.",
            ),
        )

    seconds = _wall_clock_s(m, shots, job_overhead_s)
    return CostEstimate(
        model="wall_clock",
        unit="seconds",
        value=seconds,
        breakdown={"shots": shots, "overhead_s": job_overhead_s},
        assumptions=(
            "[ASSUMPTION] no published price model for this device; reporting QPU seconds "
            "(shots × critical path + per-job overhead) instead of currency.",
        ),
    )


def _wall_clock_s(m: Mapping[str, float | int | None], shots: int, job_overhead_s: float) -> float:
    ns = m.get("critical_path_ns")
    per_shot_s = (float(ns) * 1e-9) if ns else 0.0
    return job_overhead_s + shots * per_shot_s


def circuit_metrics(circuit: ir.Circuit) -> dict[str, float | int | None]:
    """Convenience re-export so callers need only import :mod:`ptqnn.hardware.executor`."""
    return metrics(circuit)
