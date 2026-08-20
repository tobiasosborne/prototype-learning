"""OpenQASM 3 export.

Qubit indexing
--------------
The mapping is **index preserving**: our qubit ``q`` is OpenQASM ``q[q]``, and it is measured
into ``c[q]``.  What is *not* preserved is bit *significance*:

* :mod:`ptqnn.circuits.ir` puts qubit 0 in the **most** significant position of a basis index
  (``index = Σ_q b_q · 2^(n-1-q)``), so our count keys read qubit 0 first, left to right —
  the same order as the ``01101`` bit strings of ``[PTQNN Fig 5]``;
* OpenQASM/Qiskit render a classical register with ``c[n-1]`` leftmost, i.e. **``c[0]`` is the
  rightmost character**, so a QASM/Aer count key reads our qubit ``n-1`` first.

The two therefore differ by a plain string reversal, and nothing else.  Both directions are
exported: :data:`BITSTRING_ORDER` names ours, :func:`from_qasm_bitstring` and
:func:`counts_from_qasm` convert a QASM-ordered result back.  Keeping the index map trivial (and
fixing the discrepancy at the bit-string level) is the choice that makes the OpenQASM text
readable next to our circuit diagrams; the alternative — emitting ``q[n-1-i]`` — makes the text
match bit-for-bit but scrambles every qubit label.

Gate coverage
-------------
``rx ry rz h x y z sx cx cz crx swap`` come from ``stdgates.inc``.  The vendor-native names this
package introduces (``prx``/``u1q``, ``gpi``, ``gpi2``, ``rzz``, ``zz``, ``ms``, ``iswap``) are
not in ``stdgates.inc``, so a ``gate`` declaration is emitted for each one that the circuit
actually uses, written as an exact identity in terms of standard gates (see
:data:`_GATE_DEFS`).  ``from_openqasm3`` is deliberately absent: use ``qiskit.qasm3.loads``.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ptqnn.circuits import ir

BITSTRING_ORDER = "msb-first (qubit 0 leftmost)"
"""How :mod:`ptqnn.hardware.executor` count keys are ordered.  QASM/Aer keys are the reverse."""

QASM_BITSTRING_ORDER = "lsb-first (qubit n-1 leftmost, because c[0] is rightmost)"

_STD = {"rx", "ry", "rz", "h", "x", "y", "z", "sx", "cx", "cz", "crx", "swap", "s", "sdg", "id"}

# Exact definitions for the non-standard names.  Each is a *definition*, not an approximation:
#   prx(θ,φ) = RZ(φ)·RX(θ)·RZ(-φ)                       [applied right to left]
#   gpi(φ)   = [[0, e^{-iφ}], [e^{iφ}, 0]] = U(π, φ, π-φ) exactly (OpenQASM's builtin U)
#   gpi2(φ)  = prx(π/2, φ)
#   rzz(θ)   = exp(-iθ Z⊗Z/2) = CX · (I⊗RZ(θ)) · CX
#   ms(θ)    = exp(-iθ X⊗X/2) = (H⊗H) · rzz(θ) · (H⊗H)
#   iswap    = standard; expressed with s, h and two cx
_GATE_DEFS: dict[str, str] = {
    # NB the formal parameters are named ``a_theta, b_phi`` and not ``theta, phi``:
    # ``qiskit_qasm3_import`` 0.6 binds a gate's arguments in *alphabetical* order of the formal
    # names rather than declaration order, so ``(theta, phi)`` would silently come back swapped.
    # Alphabetical == declaration order here, which sidesteps the bug in both directions.
    "prx": "gate prx(a_theta, b_phi) q { rz(-b_phi) q; rx(a_theta) q; rz(b_phi) q; }",
    "u1q": "gate u1q(a_theta, b_phi) q { rz(-b_phi) q; rx(a_theta) q; rz(b_phi) q; }",
    "gpi": "gate gpi(phi) q { U(pi, phi, pi - phi) q; }",
    "gpi2": "gate gpi2(phi) q { rz(-phi) q; rx(pi/2) q; rz(phi) q; }",
    "rzz": "gate rzz(theta) a, b { cx a, b; rz(theta) b; cx a, b; }",
    "zz": "gate zz a, b { cx a, b; rz(pi/2) b; cx a, b; }",
    "ms": "gate ms(theta) a, b { h a; h b; cx a, b; rz(theta) b; cx a, b; h a; h b; }",
    "iswap": "gate iswap a, b { s a; s b; h a; cx a, b; cx b, a; h b; }",
}

# Gate definitions that use other custom definitions must come after them; ``zz``/``ms``/``rzz``
# only use stdgates, so a fixed alphabetical order is enough.
_DEF_ORDER = ("prx", "u1q", "gpi", "gpi2", "rzz", "zz", "ms", "iswap")


def _fmt(x: float) -> str:
    return f"{float(x):.17g}"


def _args(params: tuple[float, ...]) -> str:
    return f"({', '.join(_fmt(p) for p in params)})" if params else ""


def to_openqasm3(
    circuit: ir.Circuit,
    *,
    measure: bool = True,
    qreg: str = "q",
    creg: str = "c",
) -> str:
    """Serialise a unitary circuit to OpenQASM 3.

    ``measure=True`` (the default) appends a measurement of **every** qubit into ``creg[i]``;
    that is what a hardware job needs and what makes the bit-string convention above matter.
    Set ``measure=False`` for a unitary round-trip check.

    Raises on channels (a noisy IR circuit is a simulator object, not a submittable program) and
    on ``initial_state`` (arbitrary state preparation would need its own synthesis pass).
    """
    n = circuit.n_qubits
    if circuit.initial_state is not None:
        raise ValueError(
            "cannot export a circuit with an explicit initial_state; synthesise the state "
            "preparation into gates first"
        )

    used: set[str] = set()
    body: list[str] = []
    for op in circuit.ops:
        if not isinstance(op, ir.Gate):
            raise ValueError(f"cannot export channel {op.name!r} to OpenQASM 3")
        name = op.name
        if name not in _STD and name not in _GATE_DEFS:
            raise NotImplementedError(
                f"no OpenQASM 3 spelling for gate {name!r}; add one to "
                "ptqnn.hardware.qasm._GATE_DEFS"
            )
        used.add(name)
        qubits = ", ".join(f"{qreg}[{q}]" for q in op.qubits)
        body.append(f"{name}{_args(op.params)} {qubits};")

    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', ""]
    defs = [_GATE_DEFS[g] for g in _DEF_ORDER if g in used]
    if defs:
        lines += defs
        lines.append("")
    lines.append(f"qubit[{n}] {qreg};")
    if measure:
        lines.append(f"bit[{n}] {creg};")
    lines.append("")
    lines += body
    if measure:
        lines.append("")
        lines += [f"{creg}[{i}] = measure {qreg}[{i}];" for i in range(n)]
    return "\n".join(lines) + "\n"


def from_qasm_bitstring(key: str) -> str:
    """QASM/Aer key (``c[n-1]`` first) → our key (qubit 0 first).  A string reversal."""
    return key.replace(" ", "")[::-1]


def counts_from_qasm(counts: Mapping[str, int]) -> dict[str, int]:
    """Re-key a QASM/Aer counts dict into :data:`BITSTRING_ORDER`."""
    out: dict[str, int] = {}
    for k, v in counts.items():
        out[from_qasm_bitstring(k)] = out.get(from_qasm_bitstring(k), 0) + int(v)
    return out


def bit_reversal_permutation(n: int) -> np.ndarray:
    """Permutation ``R`` with ``U_ours = R · U_qiskit · R†`` for the same OpenQASM text.

    ``R`` maps the basis index whose bits are read qubit-0-first to the index whose bits are read
    qubit-``(n-1)``-first, i.e. it reverses the bits of the index.  ``R = R† = R⁻¹``.
    """
    r = np.zeros((2**n, 2**n), dtype=np.complex128)
    for i in range(2**n):
        j = int(format(i, f"0{n}b")[::-1], 2)
        r[j, i] = 1.0
    return r
