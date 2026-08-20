"""Red-phase tests for ``ptqnn.noise.model``: gate-attached channels, ASAP idle scheduling,
readout error, and the ``NoisyBackend`` wrapper.  Cross-checked against Qiskit-Aer where the
two models coincide (no idling)."""

import numpy as np
import pytest

from ptqnn.backends import reference as ref
from ptqnn.circuits import ir
from ptqnn.noise import channels as ch
from ptqnn.noise.model import NoiseModel, NoisyBackend


def test_zero_noise_model_is_identity() -> None:
    circ = ir.Circuit(2, [ir.h(0), ir.crx(0, 1, 0.4)])
    nm = NoiseModel()
    assert nm.is_trivial()
    noisy = nm.apply(circ)
    assert [o.name for o in noisy.ops] == ["h", "crx"]
    be = NoisyBackend(ref.DensityMatrixBackend(), nm)
    np.testing.assert_allclose(be.probabilities(circ), ref.StatevectorBackend().probabilities(circ))


def test_gate_channels_are_attached_after_each_gate() -> None:
    circ = ir.Circuit(2, [ir.rx(0, 0.3), ir.crx(0, 1, 0.4), ir.h(1)])
    nm = NoiseModel(one_qubit=ch.depolarizing(0.01, 1), two_qubit=ch.depolarizing(0.02, 2))
    noisy = nm.apply(circ)
    names = [o.name for o in noisy.ops]
    assert names == ["rx", "depolarizing", "crx", "depolarizing", "h", "depolarizing"]
    assert noisy.ops[3].qubits == (0, 1)
    assert isinstance(noisy.ops[3], ir.Channel) and len(noisy.ops[3].kraus) == 16


def test_per_gate_override_and_skip() -> None:
    circ = ir.Circuit(1, [ir.rx(0, 0.3), ir.h(0)])
    nm = NoiseModel(one_qubit=ch.depolarizing(0.01, 1), per_gate={"h": None})
    noisy = nm.apply(circ)
    assert [o.name for o in noisy.ops] == ["rx", "depolarizing", "h"]
    nm2 = NoiseModel(one_qubit=ch.depolarizing(0.01, 1), per_gate={"h": ch.bit_flip(0.5)})
    assert [o.name for o in nm2.apply(circ).ops] == ["rx", "depolarizing", "h", "pauli"]


def test_idle_relaxation_is_scheduled_asap() -> None:
    # rx(0) twice, then crx(0,1): qubit 1 idles for 2·d1q before the crx; no other idling.
    circ = ir.Circuit(2, [ir.rx(0, 0.1), ir.rx(0, 0.2), ir.crx(0, 1, 0.3)])
    nm = NoiseModel(t1_us=100.0, t2_us=80.0, duration_1q_ns=50.0, duration_2q_ns=300.0)
    noisy = nm.apply(circ)
    names = [(o.name, o.qubits) for o in noisy.ops]
    assert names == [
        ("rx", (0,)),
        ("relax", (0,)),
        ("rx", (0,)),
        ("relax", (0,)),
        ("idle", (1,)),
        ("crx", (0, 1)),
        ("relax", (0,)),
        ("relax", (1,)),
    ]
    idle = noisy.ops[4]
    assert isinstance(idle, ir.Channel)
    # the idle channel must equal thermal relaxation for 100 ns = 0.1 µs
    rho = np.array([[0.3, 0.4], [0.4, 0.7]], dtype=complex)
    expected = ch.apply(ch.thermal_relaxation(100.0, 80.0, 0.1), rho)
    np.testing.assert_allclose(ch.apply(idle.kraus, rho), expected, atol=1e-12)
    assert nm.schedule(circ).total_ns == pytest.approx(400.0)


def test_final_padding_before_readout() -> None:
    # qubit 1 never touched: it idles for the whole circuit before measurement.
    circ = ir.Circuit(2, [ir.rx(0, 0.1)])
    nm = NoiseModel(t1_us=10.0, t2_us=10.0, duration_1q_ns=1000.0)
    noisy = nm.apply(circ)
    assert [(o.name, o.qubits) for o in noisy.ops][-1] == ("idle", (1,))


def test_readout_error_applied_on_probabilities() -> None:
    circ = ir.Circuit(1, [])
    nm = NoiseModel(readout=ch.ReadoutError(p01=0.1, p10=0.0))
    be = NoisyBackend(ref.DensityMatrixBackend(), nm)
    np.testing.assert_allclose(be.probabilities(circ), [0.9, 0.1])
    assert be.name.startswith("noisy(")


def test_fully_depolarising_gate_gives_uniform_marginal() -> None:
    circ = ir.Circuit(2, [ir.x(0)])
    nm = NoiseModel(one_qubit=ch.depolarizing(1.0, 1))
    p = NoisyBackend(ref.DensityMatrixBackend(), nm).probabilities(circ)
    np.testing.assert_allclose(ref.marginal(p, [0]), [0.5, 0.5])
    np.testing.assert_allclose(ref.marginal(p, [1]), [1.0, 0.0])


def test_scaled_model() -> None:
    nm = NoiseModel(
        one_qubit=ch.depolarizing(0.01, 1), t1_us=100.0, t2_us=50.0, duration_1q_ns=100.0
    )
    doubled = nm.scaled(2.0)
    assert doubled.t1_us == pytest.approx(50.0)
    assert ch.average_gate_fidelity(doubled.one_qubit or []) == pytest.approx(1 - 0.02 / 2)


@pytest.mark.slow
def test_matches_qiskit_aer_without_idling() -> None:
    qiskit = pytest.importorskip("qiskit")
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel as AerNoiseModel
    from qiskit_aer.noise import depolarizing_error, thermal_relaxation_error

    t1, t2, d1, d2, p1, p2 = 80.0, 60.0, 50.0, 300.0, 0.01, 0.03
    # synchronised circuit: no idle gaps, so Aer's gate-only model equals ours.
    circ = ir.Circuit(
        2, [ir.rx(0, 0.7), ir.rx(1, 1.1), ir.crx(0, 1, 0.9), ir.rx(0, 0.3), ir.rx(1, 0.2)]
    )
    nm = NoiseModel(
        one_qubit=ch.depolarizing(p1, 1),
        two_qubit=ch.depolarizing(p2, 2),
        t1_us=t1,
        t2_us=t2,
        duration_1q_ns=d1,
        duration_2q_ns=d2,
    )
    ours = NoisyBackend(ref.DensityMatrixBackend(), nm).probabilities(circ)

    qc = qiskit.QuantumCircuit(2)
    # qiskit qubit k ↔ our qubit (n-1-k) so that bit strings agree after reversal.
    qc.rx(0.7, 1)
    qc.rx(1.1, 0)
    # Aer's density-matrix method has no native crx: use the labelled unitary so the error
    # attaches by label.  qargs[0] is qiskit's least-significant factor: [target, control].
    qc.unitary(ir.crx(0, 1, 0.9).matrix, [0, 1], label="crx")
    qc.rx(0.3, 1)
    qc.rx(0.2, 0)
    qc.save_probabilities()
    relax1 = thermal_relaxation_error(t1 * 1e3, t2 * 1e3, d1)  # Aer wants ns
    e1 = relax1.compose(depolarizing_error(p1, 1))
    relax2 = thermal_relaxation_error(t1 * 1e3, t2 * 1e3, d2)
    e2 = relax2.tensor(relax2).compose(depolarizing_error(p2, 2))
    anm = AerNoiseModel()
    anm.add_all_qubit_quantum_error(e1, ["rx"])
    anm.add_all_qubit_quantum_error(e2, ["crx"])
    sim = AerSimulator(method="density_matrix", noise_model=anm)
    res = sim.run(qc).result()
    aer = np.asarray(res.data()["probabilities"])
    np.testing.assert_allclose(ours, aer, atol=1e-9)


def test_scaled_zero_is_noiseless() -> None:
    nm = NoiseModel(one_qubit=ch.depolarizing(0.01, 1), t1_us=100.0, t2_us=50.0)
    assert nm.scaled(0.0).is_trivial()
