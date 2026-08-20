"""Vendor native gate sets — the basis a circuit must be rewritten into before submission.

Every entry records *what the vendor documents*, with the URL and a quoted fragment, and
nothing else: no fidelities, no durations, no invented aliases.  Numbers live in
``ptqnn/noise/devices/*.toml``; this module is pure gate-set vocabulary.

Naming: we use lower-case IR names throughout (``rz``, ``sx``, ``x``, ``cz``, ``u1q``, ``rzz``,
``zz``, ``gpi``, ``gpi2``, ``ms``, ``prx``, ``rx``, ``iswap``, ``ry``, ``cx``).  The vendor's own
spelling (``R_z``, ``U1q``, ``GPi2``, ``PRX``, ``Iswap``) is quoted in ``notes``.

Matrix conventions used by :mod:`ptqnn.hardware.transpile` for the non-standard names
(all in the ``ir`` convention ``R_P(θ) = exp(-iθP/2)``, first listed qubit = most significant):

* ``prx(θ, φ) = u1q(θ, φ) = RZ(φ)·RX(θ)·RZ(-φ) = exp(-i(θ/2)(cos φ·X + sin φ·Y))``
* ``gpi(φ) = [[0, e^{-iφ}], [e^{iφ}, 0]] = i·prx(π, φ)``
* ``gpi2(φ) = prx(π/2, φ)``
* ``rzz(θ) = exp(-i(θ/2)·Z⊗Z)``, ``zz = rzz(π/2)`` (Quantinuum's fully-entangling ``ZZ``)
* ``ms(θ) = exp(-i(θ/2)·X⊗X)``, fully entangling at ``θ = π/2``
* ``iswap = |00⟩⟨00| + i|01⟩⟨10| + i|10⟩⟨01| + |11⟩⟨11|``
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateSet:
    """A vendor basis: the 1q and 2q gate names a submitted circuit may contain."""

    name: str
    one_qubit: tuple[str, ...]
    two_qubit: tuple[str, ...]
    notes: str

    def supports(self, gate_name: str) -> bool:
        return gate_name in self.one_qubit or gate_name in self.two_qubit

    @property
    def entangler(self) -> str:
        """The 2q gate the transpiler targets: the first listed one it can synthesise into."""
        return self.two_qubit[0]


IBM_HERON = GateSet(
    name="ibm_heron",
    one_qubit=("rz", "sx", "x"),
    two_qubit=("cz",),
    notes=(
        "IBM Quantum docs, backend instruction list for a Heron r2 QPU (ibm_kingston): "
        "\"Supported instructions: ['cz', 'id', 'delay', 'measure', ... 'rz', 'sx', "
        "'x', ...]\".  Eagle-generation QPUs use ECR instead of CZ; the 1q basis is unchanged.  "
        "https://quantum.cloud.ibm.com/docs/guides/qpu-information (retrieved 2026-08-20)"
    ),
)

QUANTINUUM_H = GateSet(
    name="quantinuum_h",
    one_qubit=("u1q", "rz"),
    two_qubit=("zz", "rzz"),
    notes=(
        "Quantinuum H-series user guide: the native set is the arbitrary-axis xy rotation "
        'U1q(θ, φ) = R_xy, the virtual R_z, and the two-qubit ZZ / R_zz.  "The parameterized '
        "rotation around the z-axis, R_z (λ), is performed virtually within the software. All "
        'other physical gates are constructed from this set."  '
        "https://docs.quantinuum.com/systems/user_guide/hardware_user_guide/h2.html "
        "(retrieved 2026-08-20)"
    ),
)

IONQ = GateSet(
    name="ionq",
    one_qubit=("gpi", "gpi2"),
    two_qubit=("ms", "zz"),
    notes=(
        "IonQ native-gate guide: GPi, GPi2 on both Aria and Forte; the Mølmer-Sørensen MS gate "
        'is Aria-only and ZZ is Forte-only — "The physical implementation of Raman gates differs '
        "between Forte and Aria, which means they support a slightly different set of native "
        'gates."  IonQ parameterises its angles in *turns* (1 turn = 2π); our angles are radians. '
        "https://docs.ionq.com/guides/getting-started-with-native-gates (retrieved 2026-08-20)"
    ),
)

IQM = GateSet(
    name="iqm",
    one_qubit=("prx",),
    two_qubit=("cz",),
    notes=(
        'IQM (Garnet/Emerald) native operations: "the `prx` gate, the `cz` gate, the `measure` '
        "operation, the conditional PRX operation `cc_prx`, the `reset` operation, and the "
        '`barrier` operation."  There is no native RZ, so a Z rotation costs two PRX pulses.  '
        "https://docs.iqm.tech/iqm-pulse/custom_gates.html (retrieved 2026-08-20)"
    ),
)

RIGETTI_ANKAA = GateSet(
    name="rigetti_ankaa",
    one_qubit=("rx", "rz"),
    two_qubit=("cz", "iswap"),
    notes=(
        'AWS Braket device guide, Ankaa-3 verbatim mode: "Ankaa-3 supports the following native '
        "gates: 'rx', 'rz', 'iswap'. Rigetti superconducting quantum processors can run the 'rx' "
        "gate with only the angles of ±π/2 or ±π.\"  Rigetti's own compiler docs describe the "
        'older CZ-based family: "A gate is native if it is of the form `RZ(θ)` for any value `θ`, '
        '`RX(k*π/2)` for an integer `k`, or `CZ q0 q1`".  We list CZ first because '
        ":mod:`ptqnn.hardware.transpile` synthesises CZ but not iSWAP; a verbatim Ankaa-3 "
        "submission must go through Rigetti's own compiler for the iSWAP rebase.  "
        "https://docs.aws.amazon.com/braket/latest/developerguide/braket-submit-tasks.html and "
        "https://pyquil-docs.rigetti.com/en/stable/compiler.html (retrieved 2026-08-20)"
    ),
)

GENERIC = GateSet(
    name="generic",
    one_qubit=("rz", "ry", "rx"),
    two_qubit=("cx",),
    notes=(
        "Textbook basis, not a vendor's: arbitrary Euler rotations plus CNOT.  Used as the "
        "device-independent reference point for gate counts and as the target when a DeviceSpec "
        "names a 2q gate we do not recognise."
    ),
)

GATESETS: dict[str, GateSet] = {
    gs.name: gs for gs in (IBM_HERON, QUANTINUUM_H, IONQ, IQM, RIGETTI_ANKAA, GENERIC)
}


def gateset(name: str) -> GateSet:
    if name not in GATESETS:
        raise KeyError(f"unknown gate set {name!r}; available: {sorted(GATESETS)}")
    return GATESETS[name]
