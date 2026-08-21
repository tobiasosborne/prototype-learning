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


def test_idle_fidelity_limits() -> None:
    assert hp.idle_fidelity(0.0, 100.0, 80.0) == pytest.approx(1.0)
    # t ≫ T1, T2: the channel is the reset-to-|0⟩ map, F_avg = ½ (Nielsen, d = 2).
    assert hp.idle_fidelity(1e9, 100.0, 80.0) == pytest.approx(0.5)
    # ibm_fez numbers the report prints: t = 3.17 µs, T1 = 144.855 µs, T2 = 87.951 µs.
    f = hp.idle_fidelity(3168.0, 144.855, 87.951)
    assert f**7 == pytest.approx(0.8971, abs=2e-4)


def test_stateprep_dense_count_is_shende_bullock_markov() -> None:
    s = hp.section_stateprep(n=12, k_train=120)
    assert s["dense_cnots_sbm"] == 2**13 - 24 - 2 == 8166  # not 4070 = 2^n − 2n − 2
    assert s["sparse_cnots_leading_term"] == 1440
    assert 0 < s["h2_1_fidelity_dense_sbm"] < s["h2_1_fidelity_sparse"] < 1


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
