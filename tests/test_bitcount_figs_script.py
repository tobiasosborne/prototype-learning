"""Unit tests for the provenance/bookkeeping helpers of ``experiments/bitcount_figs.py``.

Review 004 B2/M8: ``final_kl`` in ``meta.json`` and the results.md table reported the best
monitored KL, not the final one, and no trained parameters were archived.  The driver is not run
end to end here; its worker runs on a one-sweep / three-step budget.
"""

import importlib.util
import os
import platform
import socket
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "bitcount_figs.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bitcount_figs", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["bitcount_figs"] = module
    spec.loader.exec_module(module)
    return module


bf = _load()


@pytest.fixture(scope="module")
def worker_result() -> dict[str, Any]:
    # seed 0; 1 superposition sweep; 3 iterative steps monitored every 2 (off-cadence final point)
    result: dict[str, Any] = bf._train_seed((0, 1, 3, "perbasis", 0, 2))
    return result


def test_worker_records_final_and_best_for_both_schemes(worker_result: dict[str, Any]) -> None:
    r = worker_result
    assert r["iterative_final"] == r["iterative_history"][-1]
    assert r["iterative_loss"] == min(r["iterative_history"])
    assert r["superposition_final"] == r["superposition_history"][-1]
    # the superposition best can come from inside a sweep, so it bounds the recorded series
    assert r["superposition_loss"] <= min(r["superposition_history"])
    assert r["iterative_iterations"] == 3
    assert len(r["iterative_history"]) == 3  # iterations 0, 2, 3


def test_worker_returns_final_and_best_parameters(worker_result: dict[str, Any]) -> None:
    net = bf.bitcount.network(bf.N_BITS, bf.N_HIDDEN)
    for key in (
        "superposition_params",
        "superposition_final_params",
        "iterative_params",
        "iterative_final_params",
    ):
        assert len(worker_result[key]) == net.n_params, key
    # the final parameters reproduce the final KL of each scheme
    final_sup = bf.bitcount.kl_loss(net)(np.array(worker_result["superposition_final_params"]))
    assert final_sup == pytest.approx(worker_result["superposition_final"], abs=1e-12)
    final_itr = bf.bitcount.perbasis_kl_loss(net)(np.array(worker_result["iterative_final_params"]))
    assert final_itr == pytest.approx(worker_result["iterative_final"], abs=1e-12)


def _fake(seed: int) -> dict[str, Any]:
    return {
        "seed": seed,
        "superposition_loss": 0.01 + seed,
        "superposition_final": 0.02 + seed,
        "iterative_loss": 0.001 + seed,
        "iterative_final": 0.009 + seed,
        "superposition_params": [float(seed), 1.0],
        "superposition_final_params": [float(seed), 2.0],
        "iterative_params": [float(seed), 3.0],
        "iterative_final_params": [float(seed), 4.0],
    }


def test_kl_tables_separate_final_from_best() -> None:
    results = [_fake(0), _fake(1)]
    final, best = bf.final_kl(results), bf.best_kl(results)
    assert final == {
        "0": {"superposition": 0.02, "iterative": 0.009},
        "1": {"superposition": 1.02, "iterative": 1.009},
    }
    assert best == {
        "0": {"superposition": 0.01, "iterative": 0.001},
        "1": {"superposition": 1.01, "iterative": 1.001},
    }


def test_results_table_has_final_and_best_columns() -> None:
    lines = bf.kl_table_lines([_fake(0)])
    assert "final" in lines[0] and "best" in lines[0]
    assert lines[2] == "| 0 | 0.020000 | 0.010000 | 0.009000 | 0.001000 |"


def test_save_params_writes_both_schemes_per_seed(tmp_path: Path) -> None:
    files = bf.save_params(tmp_path, [_fake(0), _fake(3)])
    for seed in (0, 3):
        for scheme, final_val, best_val in (("superposition", 2.0, 1.0), ("iterative", 4.0, 3.0)):
            final = np.load(tmp_path / f"params-{scheme}-{seed}.npy")
            best = np.load(tmp_path / f"params-{scheme}-best-{seed}.npy")
            np.testing.assert_array_equal(final, [seed, final_val])
            np.testing.assert_array_equal(best, [seed, best_val])
    assert sorted(files) == sorted(p.name for p in tmp_path.iterdir())


def test_provenance_records_the_machine_and_pool() -> None:
    prov = bf.provenance(pool_size=4)
    assert prov == {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "pool_size": 4,
    }


def test_elapsed_times_use_a_monotonic_clock() -> None:
    assert "time.time()" not in _SCRIPT.read_text()
