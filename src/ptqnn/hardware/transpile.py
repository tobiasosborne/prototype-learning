"""Transpilation: prototype-network circuits → vendor native gate sets and coupling maps.

Three passes, each verified numerically in ``tests/test_hardware.py`` *modulo global phase*
(a submitted circuit is measured in the computational basis, so a global phase is unobservable;
every identity below is stated up to that phase and several are in fact exact):

1. **Expansion to {1q, CX}.**  The only entangler the paper's circuits use is
   ``CRX(θ)`` ``[PTQNN Fig 1, Fig 2]``, and no vendor has it natively.  We use

       CRX(θ) = (I⊗RZ(-π/2))·CX·(I⊗RY(-θ/2))·CX·(I⊗RY(θ/2))·(I⊗RZ(π/2))            (exact)

   (matrix order; the circuit applies the right-most factor first).  Proof: on the control's
   ``|0⟩`` branch the CXs are identity and the four target rotations telescope to ``I``; on the
   ``|1⟩`` branch ``X·RY(-θ/2)·X = RY(θ/2)`` gives ``RZ(-π/2)·RY(θ)·RZ(π/2) = RX(θ)`` because
   conjugation by ``RZ(-π/2)`` sends ``Y ↦ X``.  Cost: **2 CX per CRX** — the dominant hardware
   cost of a prototype neuron, which uses one CRX per (input, hidden) pair.

2. **CX → the set's entangler.**  Composed from three exact identities:

       CX(c,t) = (I⊗H)·CZ·(I⊗H)
       CZ      ≐ (RZ(-π/2)⊗RZ(-π/2))·RZZ(π/2)                    [up to e^{iπ/4}]
       XX(θ)  = (H⊗H)·RZZ(θ)·(H⊗H)                               (so MS = XX(π/2))

   so a ZZ/RZZ set (Quantinuum) and an MS set (IonQ) both cost **one native entangler per CX**,
   and a CZ set (IBM, IQM, Rigetti) costs one CZ plus two Hadamards.  iSWAP is *not* synthesised
   (see :data:`~ptqnn.hardware.gatesets.RIGETTI_ANKAA` notes).

3. **1q re-synthesis.**  Maximal runs of consecutive 1q gates on one wire are multiplied into a
   single ``2×2`` unitary and re-emitted in the target basis from its ZYZ angles
   ``U ≐ RZ(a)·RY(b)·RZ(c)``:

   * ``rz, ry, rx`` (generic)   ``RZ(a)·RY(b)·RZ(c)``                          — 3 gates
   * ``rz, sx, x``  (IBM)       ``RZ(a)·SX·RZ(π-b)·SX·RZ(π+c)``                — 5 gates, 2 SX
   * ``rx, rz``     (Rigetti)   same ZXZXZ with ``RX(π/2)`` in place of ``SX`` — 5 gates
   * ``u1q, rz``    (Quantinuum) ``RZ(a+c)·U1q(b, π/2-c)``                     — 2 gates
   * ``prx``        (IQM)       ``PRX(π,(a+c)/2)·PRX(π,0)·PRX(b, π/2-c)``      — 3 gates
   * ``gpi, gpi2``  (IonQ)      ``GPI2(a)·GPI2(a+π-b)·GPI(0)·GPI(-t/2)``,
     ``t = a + (π-b) + (π+c)``                                                 — 4 gates

   The ZXZXZ form follows from ``RY(θ) = RX(π/2)·RZ(-θ)·RZ(π)·RX(π/2)·RZ(π)`` (up to phase);
   the ``U1q``/``PRX`` form from ``RZ(-c)·RY(b)·RZ(c) = PRX(b, π/2-c)``; the ``PRX``-only Z
   rotation from ``PRX(π,φ₁)·PRX(π,φ₂) ≐ RZ(2(φ₁-φ₂))``; and ``GPI(0)·GPI(-t/2) = RZ(t)``
   exactly.  Rotations whose angle is ``0 mod 2π`` are dropped (that costs a factor ``±1``,
   i.e. global phase).  None of these are claimed optimal — they are *fixed-form* Euler
   decompositions, which is what makes them checkable.

Routing (:func:`route`) is deliberately naive: **greedy nearest-neighbour SWAP insertion along a
BFS shortest path, no lookahead, no layout search.**  It is not optimal and is not meant to be;
it exists so that depth / 2q-count / SWAP-count comparisons across topologies are honest rather
than assuming all-to-all.  A real submission hands the circuit to the vendor's own compiler.
"""

from __future__ import annotations

import warnings
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ptqnn.circuits import ir
from ptqnn.hardware.gatesets import GATESETS, GENERIC, GateSet

Complex = npt.NDArray[np.complex128]
Coupling = list[tuple[int, int]]

_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)


# --- extra gate factories (names the IR does not ship) ----------------------------------------


def prx_matrix(theta: float, phi: float) -> Complex:
    """``RZ(φ)·RX(θ)·RZ(-φ)`` — the phased X rotation (IQM ``PRX``, Quantinuum ``U1q``)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array(
        [[c, -1j * np.exp(-1j * phi) * s], [-1j * np.exp(1j * phi) * s, c]], dtype=np.complex128
    )


def prx(q: int, theta: float, phi: float) -> ir.Gate:
    return ir.Gate((q,), prx_matrix(theta, phi), "prx", (float(theta), float(phi)))


def u1q(q: int, theta: float, phi: float) -> ir.Gate:
    """Quantinuum's ``U1q(θ, φ)`` — the same matrix as :func:`prx`, a different vendor name."""
    return ir.Gate((q,), prx_matrix(theta, phi), "u1q", (float(theta), float(phi)))


def gpi_matrix(phi: float) -> Complex:
    return np.array([[0, np.exp(-1j * phi)], [np.exp(1j * phi), 0]], dtype=np.complex128)


def gpi(q: int, phi: float) -> ir.Gate:
    """IonQ ``GPi(φ)``: a π rotation about the axis ``φ`` in the xy plane (``= i·prx(π, φ)``)."""
    return ir.Gate((q,), gpi_matrix(phi), "gpi", (float(phi),))


def gpi2(q: int, phi: float) -> ir.Gate:
    """IonQ ``GPi2(φ) = prx(π/2, φ)``."""
    return ir.Gate((q,), prx_matrix(np.pi / 2, phi), "gpi2", (float(phi),))


_SX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=np.complex128)


def sx(q: int) -> ir.Gate:
    """``√X = e^{iπ/4}·RX(π/2)`` — IBM's only physical 1q pulse."""
    return ir.Gate((q,), _SX, "sx")


def cz(a: int, b: int) -> ir.Gate:
    return ir.Gate((a, b), np.diag([1, 1, 1, -1]).astype(np.complex128), "cz")


def swap(a: int, b: int) -> ir.Gate:
    m = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128)
    return ir.Gate((a, b), m, "swap")


def iswap(a: int, b: int) -> ir.Gate:
    m = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=np.complex128)
    return ir.Gate((a, b), m, "iswap")


def rzz_matrix(theta: float) -> Complex:
    p, m = np.exp(-0.5j * theta), np.exp(0.5j * theta)
    return np.diag([p, m, m, p]).astype(np.complex128)


def rzz(a: int, b: int, theta: float) -> ir.Gate:
    """``exp(-i(θ/2)·Z⊗Z)`` — Quantinuum's arbitrary-angle ``R_zz``."""
    return ir.Gate((a, b), rzz_matrix(theta), "rzz", (float(theta),))


def zz(a: int, b: int) -> ir.Gate:
    """Quantinuum's fully entangling ``ZZ = rzz(π/2)``."""
    return ir.Gate((a, b), rzz_matrix(np.pi / 2), "zz")


def ms_matrix(theta: float) -> Complex:
    xx = np.kron(_X, _X)
    m = np.cos(theta / 2) * np.eye(4, dtype=np.complex128) - 1j * np.sin(theta / 2) * xx
    return np.asarray(m, dtype=np.complex128)


def ms(a: int, b: int, theta: float = np.pi / 2) -> ir.Gate:
    """Mølmer–Sørensen ``exp(-i(θ/2)·X⊗X)``; fully entangling at ``θ = π/2``."""
    return ir.Gate((a, b), ms_matrix(theta), "ms", (float(theta),))


# --- helpers ----------------------------------------------------------------------------------


def _wrap(theta: float) -> float:
    """Fold an angle into ``(-π, π]`` so printed circuits stay readable."""
    t = float(np.remainder(theta + np.pi, 2 * np.pi) - np.pi)
    return np.pi if t == -np.pi else t


def _is_zero_rotation(theta: float) -> bool:
    """``RZ(θ) ∝ I`` iff ``θ ≡ 0 (mod 2π)``; the residual sign is a global phase."""
    return bool(abs(_wrap(theta)) < 1e-12)


def equal_up_to_phase(a: Complex, b: Complex, atol: float = 1e-9) -> bool:
    """``a ≐ b``: equal after removing one global phase.  The only equivalence hardware sees."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    idx = np.unravel_index(int(np.argmax(np.abs(a))), a.shape)
    if abs(a[idx]) < atol:
        return bool(np.allclose(b, 0, atol=atol))
    return bool(np.allclose(a * (b[idx] / a[idx]), b, atol=atol))


def zyz_angles(u: Complex) -> tuple[float, float, float]:
    """``(a, b, c)`` with ``U ≐ RZ(a)·RY(b)·RZ(c)``.  Standard ZYZ Euler angles."""
    m = np.asarray(u, dtype=np.complex128)
    det = np.linalg.det(m)
    m = m / np.sqrt(det)  # to SU(2); the branch choice is a global phase
    b = 2.0 * np.arctan2(abs(m[1, 0]), abs(m[0, 0]))
    ap, am = np.angle(m[1, 1]), np.angle(m[1, 0])
    return _wrap(ap + am), float(b), _wrap(ap - am)


# --- (a) CRX ----------------------------------------------------------------------------------


def decompose_crx(control: int, target: int, theta: float) -> list[ir.Gate]:
    """``CRX(θ)`` into ``{rz, ry, cx}`` — 2 CX and 4 single-qubit rotations, exactly.

    Returns gates in *application* order, i.e. the reverse of the matrix product in the module
    docstring.  ``ir.crx(c, t, θ).matrix`` is reproduced exactly (not merely up to phase).
    """
    return [
        ir.rz(target, np.pi / 2),
        ir.ry(target, theta / 2),
        ir.cx(control, target),
        ir.ry(target, -theta / 2),
        ir.cx(control, target),
        ir.rz(target, -np.pi / 2),
    ]


# --- (b) gate-set rewriting -------------------------------------------------------------------


def _expand_to_cx(circuit: ir.Circuit) -> list[ir.Gate]:
    """Pass 1: every op becomes a 1q gate or a ``cx``."""
    out: list[ir.Gate] = []
    for op in circuit.ops:
        if not isinstance(op, ir.Gate):
            raise ValueError(
                f"transpile only handles unitary circuits; op {op.name!r} is a channel. "
                "Attach the noise model after transpiling."
            )
        if op.n_qubits == 1 or op.name == "cx":
            out.append(op)
        elif op.name == "crx":
            out += decompose_crx(op.qubits[0], op.qubits[1], op.params[0])
        elif op.name == "cz":
            a, b = op.qubits
            out += [ir.h(b), ir.cx(a, b), ir.h(b)]
        elif op.name == "swap":
            a, b = op.qubits
            out += [ir.cx(a, b), ir.cx(b, a), ir.cx(a, b)]
        elif op.name in ("rzz", "zz"):
            # RZZ(θ) = CX·(I⊗RZ(θ))·CX, exactly; ``zz`` is the θ = π/2 case.
            a, b = op.qubits
            theta = op.params[0] if op.params else np.pi / 2
            out += [ir.cx(a, b), ir.rz(b, theta), ir.cx(a, b)]
        elif op.name == "ms":
            # MS(θ) = XX(θ) = (H⊗H)·RZZ(θ)·(H⊗H).
            a, b = op.qubits
            theta = op.params[0] if op.params else np.pi / 2
            out += [
                ir.h(a),
                ir.h(b),
                ir.cx(a, b),
                ir.rz(b, theta),
                ir.cx(a, b),
                ir.h(a),
                ir.h(b),
            ]
        else:
            raise NotImplementedError(
                f"no expansion rule for {op.name!r} on {op.n_qubits} qubits; add one to "
                "ptqnn.hardware.transpile._expand_to_cx (no KAK synthesiser here by design)"
            )
    return out


def _cx_in_gateset(control: int, target: int, gs: GateSet) -> list[ir.Gate]:
    """Pass 2: one CX as one native entangler plus 1q glue (still in the generic 1q basis)."""
    ent = gs.entangler
    if ent == "cx":
        return [ir.cx(control, target)]
    if ent == "cz":
        return [ir.h(target), cz(control, target), ir.h(target)]
    if ent in ("zz", "rzz"):
        # CX = (I⊗H)·CZ·(I⊗H) with CZ ≐ (RZ(-π/2)⊗RZ(-π/2))·RZZ(π/2).
        core = zz(control, target) if ent == "zz" else rzz(control, target, np.pi / 2)
        return [
            ir.h(target),
            core,
            ir.rz(control, -np.pi / 2),
            ir.rz(target, -np.pi / 2),
            ir.h(target),
        ]
    if ent == "ms":
        # RZZ(π/2) = (H⊗H)·MS(π/2)·(H⊗H), then the CZ identity above.
        return [
            ir.h(target),
            ir.h(control),
            ir.h(target),
            ms(control, target, np.pi / 2),
            ir.h(control),
            ir.h(target),
            ir.rz(control, -np.pi / 2),
            ir.rz(target, -np.pi / 2),
            ir.h(target),
        ]
    raise NotImplementedError(
        f"gate set {gs.name!r} names {ent!r} as its entangler and transpile has no synthesis "
        "rule for it (iSWAP in particular needs a KAK decomposition we do not implement)"
    )


def _synth_1q(u: Complex, q: int, gs: GateSet) -> list[ir.Gate]:
    """Pass 3: a single ``2×2`` unitary as a fixed-form Euler word in the set's 1q basis."""
    if equal_up_to_phase(u, _I, atol=1e-12):
        return []
    names = set(gs.one_qubit)
    a, b, c = zyz_angles(u)

    if "sx" in names or ("rx" in names and "ry" not in names):
        # ZXZXZ: U ≐ RZ(a)·X_{π/2}·RZ(π-b)·X_{π/2}·RZ(π+c).
        def half(qq: int) -> ir.Gate:
            return sx(qq) if "sx" in names else ir.rx(qq, np.pi / 2)

        if abs(b) < 1e-12:  # a pure Z rotation needs no X pulse at all
            return [] if _is_zero_rotation(a + c) else [ir.rz(q, _wrap(a + c))]
        out: list[ir.Gate] = []
        for ang, gate in ((np.pi + c, None), (None, half), (np.pi - b, None), (None, half)):
            if gate is not None:
                out.append(gate(q))
            elif ang is not None and not _is_zero_rotation(ang):
                out.append(ir.rz(q, _wrap(ang)))
        if not _is_zero_rotation(a):
            out.append(ir.rz(q, _wrap(a)))
        return out

    if "u1q" in names:
        # U ≐ RZ(a+c)·U1q(b, π/2-c).
        out = [] if abs(b) < 1e-12 else [u1q(q, _wrap(b), _wrap(np.pi / 2 - c))]
        if not _is_zero_rotation(a + c):
            out.append(ir.rz(q, _wrap(a + c)))
        return out

    if "prx" in names:
        # No native RZ: U ≐ PRX(π,(a+c)/2)·PRX(π,0)·PRX(b, π/2-c).
        out = [] if abs(b) < 1e-12 else [prx(q, _wrap(b), _wrap(np.pi / 2 - c))]
        if not _is_zero_rotation(a + c):
            out += [prx(q, np.pi, 0.0), prx(q, np.pi, _wrap((a + c) / 2))]
        return out

    if "gpi2" in names:
        # U ≐ GPI2(a)·GPI2(a+d)·RZ(t) with d = π-b, t = a+d+e, e = π+c; RZ(t) = GPI(0)·GPI(-t/2).
        d, e = np.pi - b, np.pi + c
        if abs(b) < 1e-12:
            t = a + c
            return [] if _is_zero_rotation(t) else [gpi(q, _wrap(-t / 2)), gpi(q, 0.0)]
        t = a + d + e
        out = [] if _is_zero_rotation(t) else [gpi(q, _wrap(-t / 2)), gpi(q, 0.0)]
        return [*out, gpi2(q, _wrap(a + d)), gpi2(q, _wrap(a))]

    # Generic rz/ry/rx.
    out = []
    for ang, fac in ((c, ir.rz), (b, ir.ry), (a, ir.rz)):
        if not _is_zero_rotation(ang):
            out.append(fac(q, _wrap(ang)))
    return out


def _resynthesise_1q(gates: Sequence[ir.Gate], gs: GateSet) -> list[ir.Gate]:
    """Merge maximal 1q runs per wire, then re-emit each run in the target basis."""
    pending: dict[int, Complex] = {}
    out: list[ir.Gate] = []

    def flush(q: int) -> None:
        u = pending.pop(q, None)
        if u is not None:
            out.extend(_synth_1q(u, q, gs))

    for g in gates:
        if g.n_qubits == 1:
            q = g.qubits[0]
            pending[q] = g.matrix @ pending.get(q, _I)
        else:
            for q in g.qubits:
                flush(q)
            out.append(g)
    for q in sorted(pending):
        flush(q)
    return out


def to_gateset(circuit: ir.Circuit, gs: GateSet) -> ir.Circuit:
    """Rewrite ``circuit`` into ``gs``'s basis.  Equal to the original up to a global phase."""
    stage1 = _expand_to_cx(circuit)
    stage2: list[ir.Gate] = []
    for g in stage1:
        if g.n_qubits == 2:
            stage2 += _cx_in_gateset(g.qubits[0], g.qubits[1], gs)
        else:
            stage2.append(g)
    out = ir.Circuit(circuit.n_qubits, initial_state=circuit.initial_state)
    out.extend(_resynthesise_1q(stage2, gs))
    return out


# --- (c) coupling maps and routing ------------------------------------------------------------


def _canon(edges: Iterable[tuple[int, int]]) -> Coupling:
    seen: set[tuple[int, int]] = set()
    for a, b in edges:
        if a == b:
            continue
        seen.add((min(a, b), max(a, b)))
    return sorted(seen)


def linear(n: int) -> Coupling:
    """Nearest-neighbour chain ``0-1-…-(n-1)`` — a 1-D superconducting or trapped-ion row."""
    return _canon((i, i + 1) for i in range(n - 1))


def ring(n: int) -> Coupling:
    """Chain closed on itself; ``n ≥ 3``."""
    if n < 3:
        return linear(n)
    return _canon([*((i, i + 1) for i in range(n - 1)), (n - 1, 0)])


def grid(rows: int, cols: int) -> Coupling:
    """Square lattice, row-major indices — Google/IQM-style planar connectivity."""
    e: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if c + 1 < cols:
                e.append((i, i + 1))
            if r + 1 < rows:
                e.append((i, i + cols))
    return _canon(e)


# IBM's 27-qubit heavy-hex tile (Falcon r5.11 / one Eagle-Heron unit cell): degree-3 "vertex"
# qubits joined through degree-2 "flag" qubits.  Quoted as a topology, not as a calibration.
_HEAVY_HEX_27: Coupling = _canon(
    [
        (0, 1),
        (1, 2),
        (1, 4),
        (2, 3),
        (3, 5),
        (4, 7),
        (5, 8),
        (6, 7),
        (7, 10),
        (8, 9),
        (8, 11),
        (10, 12),
        (11, 14),
        (12, 13),
        (12, 15),
        (13, 14),
        (14, 16),
        (15, 18),
        (16, 19),
        (17, 18),
        (18, 21),
        (19, 20),
        (19, 22),
        (21, 23),
        (22, 25),
        (23, 24),
        (24, 25),
        (25, 26),
    ]
)


def _bfs_order(edges: Coupling, n_nodes: int, start: int = 0) -> list[int]:
    adj = _adjacency(edges, n_nodes)
    order, seen, q = [], {start}, deque([start])
    while q:
        v = q.popleft()
        order.append(v)
        for w in sorted(adj[v]):
            if w not in seen:
                seen.add(w)
                q.append(w)
    return order


def heavy_hex_patch(n: int) -> Coupling:
    """The first ``n`` qubits of IBM's 27-qubit heavy-hex tile, relabelled in BFS order.

    BFS relabelling guarantees the induced subgraph on ``0..n-1`` is connected for every
    ``n ≤ 27`` (each new node attaches to one already numbered).  Larger patches would need
    the full Heron lattice; we raise rather than invent one.
    """
    if n > 27:
        raise ValueError(f"heavy_hex_patch supports n ≤ 27 (one tile), got {n}")
    order = _bfs_order(_HEAVY_HEX_27, 27)
    new = {old: i for i, old in enumerate(order)}
    keep = set(order[:n])
    return _canon((new[a], new[b]) for a, b in _HEAVY_HEX_27 if a in keep and b in keep)


def all_to_all(n: int) -> Coupling:
    """Every pair — trapped-ion QCCD machines (Quantinuum, IonQ) really are like this."""
    return _canon((i, j) for i in range(n) for j in range(i + 1, n))


def _adjacency(coupling: Coupling, n: int) -> list[set[int]]:
    adj: list[set[int]] = [set() for _ in range(n)]
    for a, b in coupling:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _shortest_path(adj: Sequence[set[int]], src: int, dst: int) -> list[int]:
    prev = {src: src}
    q = deque([src])
    while q:
        v = q.popleft()
        if v == dst:
            break
        for w in sorted(adj[v]):
            if w not in prev:
                prev[w] = v
                q.append(w)
    if dst not in prev:
        raise ValueError(f"coupling map is disconnected: no path {src} → {dst}")
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    return path[::-1]


def route(
    circuit: ir.Circuit,
    coupling: Coupling,
    initial_layout: Sequence[int] | Mapping[int, int] | None = None,
) -> tuple[ir.Circuit, dict[int, int]]:
    """Insert SWAPs until every 2q gate acts on coupled physical qubits.

    Greedy and **not optimal**: for each 2q gate whose operands are not adjacent we take a BFS
    shortest path and walk the control's qubit along it one SWAP at a time.  No lookahead, no
    layout optimisation, no commutation analysis — a real vendor compiler (SABRE and friends)
    will do better, often much better on heavy-hex.

    Returns ``(routed_circuit, final_map)`` where ``final_map[logical] = physical``.  With
    ``Q_m`` the basis permutation induced by a map ``m``, the routed unitary satisfies
    ``U_routed = Q_final · U_ideal · Q_initial†`` (checked in the tests).
    """
    n_phys = max(circuit.n_qubits, 1 + max((max(e) for e in coupling), default=-1))
    adj = _adjacency(coupling, n_phys)

    if initial_layout is None:
        pos = {q: q for q in range(circuit.n_qubits)}
    elif isinstance(initial_layout, Mapping):
        pos = {int(k): int(v) for k, v in initial_layout.items()}
    else:
        pos = {q: int(p) for q, p in enumerate(initial_layout)}
    if sorted(pos) != list(range(circuit.n_qubits)):
        raise ValueError("initial_layout must give a physical qubit for every logical qubit")
    if len(set(pos.values())) != len(pos):
        raise ValueError("initial_layout must be injective")

    occupant = {p: q for q, p in pos.items()}

    # Finding 3: an Iris-style circuit carries its whole dataset superposition in
    # ``initial_state`` [PTQNN §III.C].  The routed program runs on *physical* wires, so the
    # state it starts from is the logical one relabelled by the **initial** layout (the final
    # map is irrelevant here — the SWAPs that produce it have not happened yet).  Extra physical
    # wires the coupling map has but the circuit does not are appended in |0⟩; because qubit 0
    # is the most significant bit, that is a left shift of the basis index by the padding width.
    state = circuit.initial_state
    if state is not None:
        pad = np.zeros(2**n_phys, dtype=np.complex128)
        shift = n_phys - circuit.n_qubits
        pad[np.arange(2**circuit.n_qubits) << shift] = state
        free = [p for p in range(n_phys) if p not in occupant]
        full = {**pos, **dict(zip(range(circuit.n_qubits, n_phys), free, strict=True))}
        state = permutation_unitary(full, n_phys) @ pad
    out = ir.Circuit(n_phys, initial_state=state)

    def do_swap(pa: int, pb: int) -> None:
        out.append(swap(pa, pb))
        qa, qb = occupant.get(pa), occupant.get(pb)
        occupant[pa], occupant[pb] = qb, qa  # type: ignore[assignment]
        if qa is not None:
            pos[qa] = pb
        if qb is not None:
            pos[qb] = pa

    for op in circuit.ops:
        if not isinstance(op, ir.Gate):
            raise ValueError(f"route only handles unitary circuits; {op.name!r} is a channel")
        if op.n_qubits == 1:
            out.append(ir.Gate((pos[op.qubits[0]],), op.matrix, op.name, op.params))
            continue
        if op.n_qubits != 2:
            raise NotImplementedError("routing is implemented for 1- and 2-qubit gates only")
        a, b = op.qubits
        if pos[b] not in adj[pos[a]]:
            path = _shortest_path(adj, pos[a], pos[b])
            for i in range(len(path) - 2):
                do_swap(path[i], path[i + 1])
        out.append(ir.Gate((pos[a], pos[b]), op.matrix, op.name, op.params))

    return out, dict(pos)


def permutation_unitary(qubit_map: Mapping[int, int], n: int) -> Complex:
    """Basis permutation ``Q`` with ``Q|x⟩_logical = |y⟩_physical``, ``y[map[l]] = x[l]``.

    Uses the IR's convention that qubit 0 is the most significant bit.
    """
    q = np.zeros((2**n, 2**n), dtype=np.complex128)
    for xi in range(2**n):
        bits = [(xi >> (n - 1 - l)) & 1 for l in range(n)]
        yi = 0
        for l, bit in enumerate(bits):
            yi |= bit << (n - 1 - qubit_map[l])
        q[yi, xi] = 1.0
    return q


# --- (d) metrics ------------------------------------------------------------------------------


def metrics(
    circuit: ir.Circuit, durations_ns: Mapping[str, float] | None = None
) -> dict[str, float | int | None]:
    """Depth, gate counts, and (given durations) the critical-path time in ns.

    ``depth`` is the as-soon-as-possible layer count: one clock per wire, a gate starts when all
    its wires are free.  ``durations_ns`` is looked up by gate name first, then by the fallback
    keys ``"1q"``/``"2q"``; ``"swap"`` defaults to three 2q gates if not given explicitly.
    """
    n = circuit.n_qubits
    layer = [0] * n
    clock = [0.0] * n
    counts = {"n_1q": 0, "n_2q": 0, "n_swap": 0}
    have_times = durations_ns is not None
    d: Mapping[str, float] = durations_ns or {}

    for op in circuit.ops:
        qs = op.qubits
        k = len(qs)
        if k == 1:
            counts["n_1q"] += 1
        elif k == 2:
            counts["n_2q"] += 1
            if op.name == "swap":
                counts["n_swap"] += 1
        start = max((layer[q] for q in qs), default=0)
        for q in qs:
            layer[q] = start + 1
        if have_times:
            if op.name in d:
                dt = float(d[op.name])
            elif op.name == "swap":
                dt = 3.0 * float(d.get("2q", 0.0))
            else:
                dt = float(d.get(f"{k}q", 0.0))
            t0 = max((clock[q] for q in qs), default=0.0)
            for q in qs:
                clock[q] = t0 + dt

    return {
        "depth": max(layer, default=0),
        "n_1q": counts["n_1q"],
        "n_2q": counts["n_2q"],
        "n_swap": counts["n_swap"],
        "n_qubits": n,
        "critical_path_ns": (max(clock, default=0.0) if have_times else None),
    }


# --- (e) device-level compilation -------------------------------------------------------------


@dataclass(frozen=True)
class CompiledCircuit:
    """A circuit rewritten and routed for one device, with its cost metrics."""

    circuit: ir.Circuit
    qubit_map: dict[int, int]
    metrics: dict[str, float | int | None]
    gateset: GateSet
    coupling: Coupling
    device_name: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)


def gateset_for(
    native_2q: Sequence[str],
    native_1q: Sequence[str] = (),
    *,
    device_name: str = "",
    strict: bool = False,
) -> GateSet:
    """Pick a :class:`GateSet` from a device's advertised native gates.

    ``cz`` alone is ambiguous (IBM, IQM and Rigetti all use it), so the 1q set breaks the tie.
    When no rule matches — Google's ``xy_microwave``/``virtual_z`` is the shipped example — we do
    **not** quietly hand back IBM's ``{rz, sx, x}``: :func:`gateset_for` warns (or raises with
    ``strict=True``) and returns :data:`~ptqnn.hardware.gatesets.GENERIC`, whose notes say it is
    not a vendor basis.  Use :func:`gateset_note_for` to get the caveat as a string; that is what
    :func:`compile_for` puts into ``CompiledCircuit.notes`` so a report cannot lose it.
    """
    gset, note = gateset_note_for(native_2q, native_1q, device_name=device_name)
    if note is not None:
        if strict:
            raise ValueError(note)
        warnings.warn(note, RuntimeWarning, stacklevel=2)
    return gset


def gateset_note_for(
    native_2q: Sequence[str],
    native_1q: Sequence[str] = (),
    *,
    device_name: str = "",
) -> tuple[GateSet, str | None]:
    """``(gate set, caveat or None)``.  The caveat is an ``[ASSUMPTION]`` line, ready to log."""
    two = {g.lower() for g in native_2q}
    one = {g.lower() for g in native_1q}
    who = device_name or f"native_1q={sorted(one)}, native_2q={sorted(two)}"

    if two & {"ms", "xx"}:
        return GATESETS["ionq"], None
    if two & {"zz", "rzz", "zzphase", "zzmax"}:
        if one & {"gpi", "gpi2"}:
            return GATESETS["ionq"], None
        return GATESETS["quantinuum_h"], None
    if "cz" in two or "iswap" in two:
        if one & {"prx"}:
            return GATESETS["iqm"], None
        if one & {"sx"}:
            return GATESETS["ibm_heron"], None
        if one <= {"rx", "rz"} and one:
            note = None
            if "cz" not in two:
                # Ankaa-3's verbatim set is {rx, rz, iswap}; we synthesise CZ, not iSWAP.
                note = (
                    f"[ASSUMPTION] {who} advertises no CZ: transpiled to CZ anyway because "
                    "ptqnn.hardware.transpile has no iSWAP synthesis.  A verbatim submission "
                    "must be rebased by the vendor compiler (see gatesets.RIGETTI_ANKAA.notes)."
                )
            return GATESETS["rigetti_ankaa"], note
        return GENERIC, (
            f"[ASSUMPTION] {who} names a CZ/iSWAP entangler but a 1q basis we have no Euler "
            "word for; compiled into the GENERIC {rz, ry, rx; cx} basis, which is NOT this "
            "device's native set — gate counts are indicative only."
        )
    if two & {"cx", "cnot", "ecr"}:
        return GENERIC, None
    return GENERIC, (
        f"[ASSUMPTION] {who} names no recognised entangler; compiled into the GENERIC "
        "{rz, ry, rx; cx} basis, which is NOT this device's native set."
    )


def coupling_for(topology: str, n: int) -> tuple[Coupling, str]:
    """Coupling map for a topology string, plus a note recording what we assumed."""
    t = topology.strip().lower()
    if t in ("all-to-all", "all to all", "alltoall", "full", "complete"):
        return all_to_all(n), "all-to-all as published"
    if "heavy" in t:
        return heavy_hex_patch(n), "[ASSUMPTION] first n qubits of one 27-qubit heavy-hex tile"
    if "linear" in t or "chain" in t:
        return linear(n), "linear chain as published"
    if "ring" in t or "cycl" in t:
        return ring(n), "ring as published"
    if "grid" in t or "square" in t or "lattice" in t:
        side = int(np.ceil(np.sqrt(n)))
        # Keep the row-major prefix of length n: it is connected (node i>0 always couples to
        # i-1 or i-side, both already in the prefix) and gives a coupling map of exactly n nodes.
        patch = _canon((a, b) for a, b in grid(side, side) if a < n and b < n)
        return patch, f"[ASSUMPTION] {side}×{side} square-lattice patch, first {n} sites"
    return all_to_all(n), f"[ASSUMPTION] topology {topology!r} unrecognised; assumed all-to-all"


def compile_for(
    circuit: ir.Circuit,
    device: object,
    layout: Sequence[int] | Mapping[int, int] | None = None,
) -> CompiledCircuit:
    """Rewrite ``circuit`` for ``device`` (a :class:`~ptqnn.noise.presets.DeviceSpec`).

    Gate set from ``device.native_2q``/``native_1q``, coupling from ``device.topology``.  Routing
    runs *before* the basis rewrite — a ``CRX`` and the two ``CX`` gates it becomes make exactly the
    same adjacency demand, so routing early is free and rebasing once afterwards means the
    router's SWAPs are themselves emitted natively.  Each SWAP costs 3 entanglers, which is why
    the linear and heavy-hex rows of ``reports/hardware-compile-table.md`` are several times the
    all-to-all ones: the prototype network is a complete bipartite graph, the worst possible fit
    for a lattice.  ``metrics["n_swap"]`` is the router's count (the rebase has by then dissolved
    the SWAP gates); ``metrics["n_2q_from_swap"] = 3·n_swap`` is what they cost.
    """
    n = circuit.n_qubits
    dev_n = int(getattr(device, "n_qubits", 0) or 0)
    if dev_n and n > dev_n:
        raise ValueError(
            f"circuit needs {n} qubits, device {getattr(device, 'name', '?')} has {dev_n}"
        )

    gs, gs_note = gateset_note_for(
        getattr(device, "native_2q", ()),
        getattr(device, "native_1q", ()),
        device_name=str(getattr(device, "name", "")),
    )
    coupling, note = coupling_for(str(getattr(device, "topology", "")), n)

    # Route the logical circuit first (a CRX and its two CXs have the same adjacency demand),
    # then rebase once — so the SWAPs the router inserts are themselves rewritten natively.
    routed, qmap = route(circuit, coupling, layout)
    final = to_gateset(routed, gs)

    d1 = float(getattr(device, "gate_1q_ns", 0.0) or 0.0)
    d2 = float(getattr(device, "gate_2q_ns", 0.0) or 0.0)
    durations = {"1q": d1, "2q": d2} if (d1 or d2) else None
    notes = [note]
    if gs_note is not None:
        notes.append(gs_note)
    if durations is None:
        notes.append("[ASSUMPTION] no published gate durations: critical_path_ns is undefined")
    if not float(getattr(device, "readout_ns", 0.0) or 0.0):
        notes.append(
            "[ASSUMPTION] no published readout duration: critical_path_ns omits measurement, "
            "which on a superconducting QPU is the longest single operation in the program"
        )

    # After the rebase every SWAP has become three native entanglers, so ``metrics(final)`` can
    # no longer see them; carry the router's own count across and record what it cost.
    m = metrics(final, durations)
    n_swap = routed.count("swap")
    m["n_swap"] = n_swap
    m["n_2q_from_swap"] = 3 * n_swap

    # Finding 4: every shot ends in a measurement, and on a superconducting QPU that readout is
    # the longest single operation in the program (IBM Fez: 1560 ns readout vs ~1056 ns of
    # gates).  ``metrics`` only walks ``circuit.ops``, so add it here and report it separately.
    # It stays out of the *noise* model on purpose: the vendor's asymmetric p10 already IS the
    # T1 decay during readout (1 - exp(-1560 ns / 144.855 µs) = 0.0107 vs p10 = 0.0115), so an
    # extra relaxation channel before the confusion map would double count it.
    ro = float(getattr(device, "readout_ns", 0.0) or 0.0)
    m["gate_time_ns"] = m["critical_path_ns"]
    m["readout_ns"] = ro or None
    if m["critical_path_ns"] is not None:
        m["critical_path_ns"] = float(m["critical_path_ns"]) + ro

    return CompiledCircuit(
        circuit=final,
        qubit_map=qmap,
        metrics=m,
        gateset=gs,
        coupling=coupling,
        device_name=str(getattr(device, "name", "")),
        notes=tuple(notes),
    )
