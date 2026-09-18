"""Regression tests for the pure arithmetic of ``experiments/hardware_plan.py`` — the numbers
review 002 blocker 1 found wrong in ``reports/hardware-plan.md`` (state-prep CNOT count, SWAP
share, idle fidelity, fidelity budget factors)."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from ptqnn.noise import presets

_SPEC = importlib.util.spec_from_file_location(
    "hardware_plan", Path(__file__).resolve().parents[1] / "experiments" / "hardware_plan.py"
)
assert _SPEC is not None and _SPEC.loader is not None
hp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hp)


_IRIS_RUNS_PRESENT = all(
    (Path(__file__).resolve().parents[1] / "experiments/runs/iris" / d).is_dir()
    for d in ("20260820T144849Z", "20260820T145454Z", "20260820T145455Z")
)
_needs_iris_runs = pytest.mark.skipif(
    not _IRIS_RUNS_PRESENT,
    reason="reads the pinned Iris run dirs (gitignored; present on the workstation only)",
)


def test_idle_fidelity_limits() -> None:
    assert hp.idle_fidelity(0.0, 100.0, 80.0) == pytest.approx(1.0)
    # t ≫ T1, T2: the channel is the reset-to-|0⟩ map, F_avg = ½ (Nielsen, d = 2).
    assert hp.idle_fidelity(1e9, 100.0, 80.0) == pytest.approx(0.5)
    # ibm_fez numbers the report prints: t = 3.17 µs, T1 = 144.855 µs, T2 = 87.951 µs.
    f = hp.idle_fidelity(3168.0, 144.855, 87.951)
    assert f**7 == pytest.approx(0.8971, abs=2e-4)


@_needs_iris_runs
def test_stateprep_dense_count_is_shende_bullock_markov() -> None:
    s = hp.section_stateprep(n=12)
    assert s["dense_cnots_sbm"] == 2**13 - 24 == 8168  # 2^(n+1) − 2n; not 4070 = 2^n − 2n − 2
    assert 0 < s["h2_1_fidelity_dense_sbm"] < min(s["h2_1_fidelity_sparse_leading_term_per_run"])


def test_fidelity_budget_factors_multiply_and_flag_missing_fields() -> None:
    h2 = presets.device("quantinuum_h2_1")
    r = hp.fidelity_budget(h2, n_1q=87, n_2q=24, depth=36, t_gate_ns=None, nq=7)
    assert r["F_2q"] == pytest.approx((1 - 0.0011) ** 24)
    assert r["F_idle"] == pytest.approx((1 - 2e-4) ** (7 * 36))
    assert r["F_total"] == pytest.approx(r["F_2q"] * r["F_1q"] * r["F_idle"] * r["F_readout"])
    assert not r["upper_bound_only"]

    rig = presets.device("rigetti_ankaa_3", allow_missing=True)
    r = hp.fidelity_budget(rig, n_1q=100, n_2q=63, depth=218, t_gate_ns=1e4, nq=7)
    assert r["F_1q"] is None and r["F_readout"] is None and r["F_idle"] is None
    assert r["upper_bound_only"]
    assert r["F_total"] == pytest.approx(0.99**63)


def test_memory_error_read_from_provenance() -> None:
    assert hp.memory_error(presets.device("quantinuum_h2_1")) == pytest.approx(2.0e-4)
    assert hp.memory_error(presets.device("quantinuum_h1_1")) == pytest.approx(2.2e-4)
    assert hp.memory_error(presets.device("ibm_fez")) is None


def test_tvd_is_half_l1() -> None:
    assert hp.tvd(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)
    assert math.isclose(hp.tvd(np.array([0.5, 0.5]), np.array([0.5, 0.5])), 0.0)


def test_every_calibration_number_used_has_a_provenance_entry() -> None:
    """Ground truth: each shipped device TOML must carry a ``[[provenance]]`` entry for every
    non-trivial field the hardware plan reads; an unsourced number fails the build."""
    devices = {n: presets.device(n, allow_missing=True) for n in presets.available_devices()}
    pv = hp.section_provenance(devices)
    gaps = {n: p["unsourced"] for n, p in pv.items() if p["unsourced"]}
    assert gaps == {}, gaps
    # and a sourced field really points at a URL with a quote
    fez = pv["ibm_fez"]["fields"]["gate_2q_error"]["sources"][0]
    assert fez["url"].startswith("http") and fez["quote"]


def test_provenance_validator_compares_values_and_rejects_wildcards() -> None:
    """Review 003 #2: a changed calibration value, or an entry that only names a different field,
    must leave the field UNSOURCED.  The validator is exercised on a mutated IBM preset."""
    import dataclasses

    fez = presets.device("ibm_fez")
    # 1. the shipped preset is fully sourced
    assert hp.section_provenance({"ibm_fez": fez})["ibm_fez"]["unsourced"] == []
    # 2. mutate a value: the entry's value no longer matches → unsourced (the validator reads the
    #    raw preset for values, so we mutate the provenance instead: drop every entry that names
    #    gate_2q.error_median and keep the rest, including the 'snapshot epoch' wildcard)
    keep = tuple(e for e in fez.provenance if "gate_2q.error_median" not in str(e.get("field", "")))
    mutated = dataclasses.replace(fez, provenance=keep)
    import unittest.mock as um

    with um.patch.object(hp.presets, "device", return_value=mutated):
        pv = hp.section_provenance({"ibm_fez": mutated})["ibm_fez"]
    assert "gate_2q_error" in pv["unsourced"]
    # 3. an entry with the right id but a different value does not vouch
    wrong = tuple(
        {**e, "value": 0.123}
        if str(e.get("field", "")).startswith("noise.gate_1q.error_median (")
        else e
        for e in keep
    )
    wrong = tuple(
        e
        for e in wrong
        if str(e.get("field", "")) != "noise.gate_1q.error_median, noise.gate_1q.duration_ns"
    )
    with um.patch.object(
        hp.presets, "device", return_value=dataclasses.replace(fez, provenance=wrong)
    ):
        pv = hp.section_provenance({"ibm_fez": fez})["ibm_fez"]
    assert "gate_1q_error" in pv["unsourced"]


def test_entry_values_parses_combined_fields() -> None:
    e = {"field": "noise.readout.error_median, p01, p10, duration_ns", "value": [1, 2, 3, 4]}
    v = hp._entry_values(e)
    assert v["noise.readout.p01"] == 2 and v["noise.readout.duration_ns"] == 4
    e = {"field": "noise.t1_us.median (derived: median of 156 values)", "value": 144.855}
    assert hp._entry_values(e) == {"noise.t1_us.median": 144.855}
    e = {"field": "device.native_1q, device.native_2q", "value": ["cz", "rz", "sx", "x"]}
    assert hp._entry_values(e)["device.native_1q"] == ["cz", "rz", "sx", "x"]
    assert hp._entry_values({"field": "snapshot epoch", "value": "2025"}) == {}


@_needs_iris_runs
def test_stateprep_uses_distinct_training_codes() -> None:
    s = hp.section_stateprep(n=12)
    assert s["k_distinct_train_codes_per_run"] == [14, 12, 14]
    assert s["dense_cnots_sbm"] == 2**13 - 24 == 8168
    assert s["sparse_cnots_leading_term_per_run"] == [168, 144, 168]
    assert all(0.8 < f < 0.9 for f in s["h2_1_fidelity_sparse_leading_term_per_run"])
