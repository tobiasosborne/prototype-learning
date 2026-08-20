"""Red-phase tests for vendor device presets (``ptqnn.noise.presets``)."""

from pathlib import Path

import numpy as np
import pytest

from ptqnn.noise import channels as ch
from ptqnn.noise import presets

FIXTURE = """
[device]
name = "toy_qpu"
vendor = "ACME"
technology = "test"
n_qubits = 5
topology = "linear"
native_1q = ["rx", "rz"]
native_2q = ["cz"]
retrieved_at = 2026-08-20

[noise]
t1_us = {median = 100.0}
t2_us = {median = 80.0, kind = "echo"}
gate_1q = {error_median = 0.001, duration_ns = 40.0}
gate_2q = {error_median = 0.01, duration_ns = 300.0}
readout = {error_median = 0.02, p01 = 0.015, p10 = 0.025, duration_ns = 1000.0}
notes = "fixture"

[[provenance]]
field = "noise.t1_us.median"
value = 100.0
source_url = "https://example.org"
source_kind = "vendor_docs"
retrieved_at = 2026-08-20
quote = "T1 median 100 us"
"""


@pytest.fixture
def toy(tmp_path: Path) -> presets.DeviceSpec:
    p = tmp_path / "toy.toml"
    p.write_text(FIXTURE)
    return presets.load_device(p)


def test_device_spec_fields(toy: presets.DeviceSpec) -> None:
    assert toy.name == "toy_qpu"
    assert toy.n_qubits == 5
    assert toy.t1_us == 100.0 and toy.t2_us == 80.0
    assert toy.gate_1q_error == 0.001 and toy.gate_1q_ns == 40.0
    assert toy.gate_2q_error == 0.01 and toy.gate_2q_ns == 300.0
    assert toy.readout_p01 == 0.015 and toy.readout_p10 == 0.025
    assert toy.provenance[0]["field"] == "noise.t1_us.median"


def test_noise_model_from_device(toy: presets.DeviceSpec) -> None:
    nm = toy.noise_model()
    assert nm.name == "toy_qpu"
    assert nm.t1_us == 100.0 and nm.duration_2q_ns == 300.0
    # average error of the depolarising part equals the quoted gate error
    assert nm.one_qubit is not None and nm.two_qubit is not None
    assert 1 - ch.average_gate_fidelity(nm.one_qubit) == pytest.approx(0.001)
    assert 1 - ch.average_gate_fidelity(nm.two_qubit) == pytest.approx(0.01)
    assert nm.readout is not None
    np.testing.assert_allclose(nm.readout.confusion, [[0.985, 0.025], [0.015, 0.975]])


def test_readout_symmetric_fallback(tmp_path: Path) -> None:
    txt = FIXTURE.replace(
        "readout = {error_median = 0.02, p01 = 0.015, p10 = 0.025, duration_ns = 1000.0}",
        "readout = {error_median = 0.02}",
    )
    (tmp_path / "t.toml").write_text(txt)
    spec = presets.load_device(tmp_path / "t.toml")
    assert spec.readout_p01 == 0.02 and spec.readout_p10 == 0.02


def test_missing_relaxation_is_allowed(tmp_path: Path) -> None:
    txt = FIXTURE.replace("t1_us = {median = 100.0}\n", "").replace(
        't2_us = {median = 80.0, kind = "echo"}\n', ""
    )
    (tmp_path / "t.toml").write_text(txt)
    spec = presets.load_device(tmp_path / "t.toml")
    assert spec.t1_us is None
    nm = spec.noise_model()
    assert nm.t1_us is None and nm.one_qubit is not None


def test_bundled_devices_load_and_are_physical() -> None:
    names = presets.available_devices()
    if not names:
        pytest.skip("no bundled device files yet")
    for name in names:
        spec = presets.device(name, allow_missing=True)
        nm = spec.noise_model()
        assert not nm.is_trivial(), name
        if spec.t1_us is not None and spec.t2_us is not None:
            assert spec.t2_us <= 2 * spec.t1_us + 1e-9, name
        assert 0 <= spec.gate_2q_error < 0.2, name
        assert spec.provenance, f"{name} has no provenance"


def test_missing_errors_raise_unless_allowed(tmp_path: Path) -> None:
    txt = FIXTURE.replace("gate_1q = {error_median = 0.001, duration_ns = 40.0}\n", "")
    (tmp_path / "t.toml").write_text(txt)
    with pytest.raises(ValueError, match="allow_missing"):
        presets.load_device(tmp_path / "t.toml")
    spec = presets.load_device(tmp_path / "t.toml", allow_missing=True)
    assert spec.missing_fields == ("gate_1q",)
