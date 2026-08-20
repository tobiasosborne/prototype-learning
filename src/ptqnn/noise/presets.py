"""Vendor device presets: TOML calibration summaries → :class:`~ptqnn.noise.model.NoiseModel`.

Device files live in ``src/ptqnn/noise/devices/<slug>.toml`` and carry per-field provenance
(URL, retrieval date, quoted sentence) in ``[[provenance]]`` tables — a cut-down version of
``../QuantumHardware.jl/schema/SCHEMA.md``.  See ``devices/README.md`` for what is sourced and
what is missing.  Mapping to a noise model (``[ASSUMPTION]``, standard practice, cf. Aer's
``NoiseModel.from_backend``):

* 1q/2q gate error ``ε`` (= 1 − average fidelity) → depolarising channel with ``F_avg = 1 − ε``;
* median ``T1``/``T2`` + gate durations → thermal relaxation during gates and idling;
* readout: ``p01``/``p10`` when published, else the symmetric ``error_median`` for both.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from ptqnn.noise import channels as ch
from ptqnn.noise.model import NoiseModel


@dataclass(frozen=True)
class DeviceSpec:
    name: str
    vendor: str
    technology: str
    n_qubits: int
    topology: str
    native_1q: tuple[str, ...]
    native_2q: tuple[str, ...]
    t1_us: float | None
    t2_us: float | None
    gate_1q_error: float
    gate_1q_ns: float
    gate_2q_error: float
    gate_2q_ns: float
    readout_p01: float
    readout_p10: float
    readout_ns: float
    notes: str = ""
    provenance: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    missing_fields: tuple[str, ...] = ()

    def noise_model(self) -> NoiseModel:
        return NoiseModel(
            one_qubit=ch.depolarizing_from_fidelity(1 - self.gate_1q_error, 1),
            two_qubit=ch.depolarizing_from_fidelity(1 - self.gate_2q_error, 2),
            t1_us=self.t1_us,
            t2_us=self.t2_us,
            duration_1q_ns=self.gate_1q_ns,
            duration_2q_ns=self.gate_2q_ns,
            readout=ch.ReadoutError(self.readout_p01, self.readout_p10),
            name=self.name,
        )


def _median(section: dict[str, Any], key: str) -> float | None:
    v = section.get(key)
    if v is None:
        return None
    if isinstance(v, dict):
        return float(v["median"]) if "median" in v else None
    return float(v)


def parse_device(doc: dict[str, Any], allow_missing: bool = False) -> DeviceSpec:
    dev, noise = doc["device"], doc["noise"]
    g1, g2 = noise.get("gate_1q", {}), noise.get("gate_2q", {})
    ro = noise.get("readout", {})
    sym = float(ro.get("error_median", 0.0))
    t1, t2 = _median(noise, "t1_us"), _median(noise, "t2_us")
    if (t1 is None) != (t2 is None):
        # One of the two missing: drop relaxation entirely rather than guess.
        t1 = t2 = None
    missing = [k for k, sec in (("gate_1q", g1), ("gate_2q", g2)) if "error_median" not in sec]
    if "error_median" not in ro and "p01" not in ro:
        missing.append("readout")
    if missing and not allow_missing:
        raise ValueError(
            f"{dev['name']}: no published {missing}; pass allow_missing=True to model them as "
            "error-free (reviews/002 #5) — the noise model will then be optimistic"
        )
    return DeviceSpec(
        name=dev["name"],
        vendor=dev.get("vendor", ""),
        technology=dev.get("technology", ""),
        n_qubits=int(dev.get("n_qubits", 0)),
        topology=dev.get("topology", ""),
        native_1q=tuple(dev.get("native_1q", ())),
        native_2q=tuple(dev.get("native_2q", ())),
        t1_us=t1,
        t2_us=t2,
        gate_1q_error=float(g1.get("error_median", 0.0)),
        gate_1q_ns=float(g1.get("duration_ns", 0.0)),
        gate_2q_error=float(g2.get("error_median", 0.0)),
        gate_2q_ns=float(g2.get("duration_ns", 0.0)),
        readout_p01=float(ro.get("p01", sym)),
        readout_p10=float(ro.get("p10", sym)),
        readout_ns=float(ro.get("duration_ns", 0.0)),
        notes=str(noise.get("notes", "")),
        provenance=tuple(doc.get("provenance", ())),
        missing_fields=tuple(missing),
    )


def load_device(path: str | Path, allow_missing: bool = False) -> DeviceSpec:
    with open(path, "rb") as f:
        return parse_device(tomllib.load(f), allow_missing)


def _devices_dir() -> Path:
    return Path(str(resources.files("ptqnn.noise").joinpath("devices")))


def available_devices() -> list[str]:
    d = _devices_dir()
    return sorted(p.stem for p in d.glob("*.toml")) if d.is_dir() else []


def device(name: str, allow_missing: bool = False) -> DeviceSpec:
    """Bundled preset by slug.  Devices with unpublished 1q/readout errors (Rigetti Ankaa-3)
    raise unless ``allow_missing=True``, so an optimistic model is never built by accident."""
    p = _devices_dir() / f"{name}.toml"
    if not p.exists():
        raise KeyError(f"unknown device {name!r}; available: {available_devices()}")
    return load_device(p, allow_missing)
