"""Unit tests for ``experiments/iris_sweep_summary.py`` (HANDOFF item 2: the Fig 9 verdict).

The verdict rules are the ones stated in ``iris.train.firing_placement`` and
``iris.train.prototype_placement``; here they are exercised on synthetic run directories whose
answer is known, so the aggregation cannot silently invert a comparison or mix the two policies.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "iris_sweep_summary.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("iris_sweep_summary", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["iris_sweep_summary"] = module
    spec.loader.exec_module(module)
    return module


sw = _load()


def _acc(test: float, unfiltered: float) -> dict[str, Any]:
    return {"train": 1.0, "test": test, "test_unfiltered": unfiltered, "n_test": 90}


def _placement(rho_p: float, rho_c: float) -> dict[str, Any]:
    return {
        "prototype-inverse": {"rho": rho_p, "all_dead": False},
        "prototype-direct": {"rho": rho_p + 0.1, "all_dead": False},
        "classic-inverse": {"rho": rho_c, "all_dead": False},
        "classic-direct": {"rho": rho_c, "all_dead": False},
    }


def _firing(cov_p: float, sel_p: float, cov_c: float, sel_c: float) -> dict[str, Any]:
    return {
        "prototype": {"rho_f": cov_p, "rho_f_selectivity": sel_p, "n_draws": 1000},
        "classic": {"rho_f": cov_c, "rho_f_selectivity": sel_c, "n_draws": 1000},
    }


def _write_run(
    root: Path, seed: int, policy: str, *, proto_wins: bool, git_hash: str = "abc"
) -> Path:
    d = root / f"20260925T000000Z-seed{seed}-{policy}"
    d.mkdir(parents=True)
    s = 1.0 if proto_wins else -1.0
    results = {
        "accuracy": {
            "prototype": _acc(0.96 + 0.001 * seed, 0.94),
            "classic": _acc(0.95, 0.93 - 0.001 * seed),
        },
        "placement": _placement(0.6 * s + 0.01 * seed, -0.2),
        "firing": _firing(1.0 + s * 3 + 0.1 * seed, 20.0 * s + seed, 0.5, 0.2),
    }
    meta = {"seed": seed, "git_hash": git_hash, "split": {"ambiguous": policy}}
    # classic test KL rises after its minimum; prototype's is monotone
    losses = {
        "classic": {"train": [0.3, 0.01, 0.0], "test": [0.3, 0.02, 0.05]},
        "prototype": {"train": [0.3, 0.02, 0.01], "test": [0.3, 0.03, 0.02]},
    }
    for name, obj in (("results", results), ("meta", meta), ("losses", losses)):
        (d / f"{name}.json").write_text(json.dumps(obj))
    return d


def test_prototype_dominance_gives_a_positive_verdict_per_policy(tmp_path: Path) -> None:
    for seed in range(20):
        _write_run(tmp_path, seed, "merge", proto_wins=True)
        _write_run(tmp_path, seed, "drop", proto_wins=False)
    runs = sw.load_runs(sorted(tmp_path.glob("*-seed*-*")))
    for r in runs:  # what --calibrated attaches (computed from the saved parameters)
        r["calibrated"] = {k: {"test": 0.97, "test_unfiltered": 0.95} for k in sw.KINDS}
    summary = sw.aggregate(runs, n_boot=2000, boot_seed=0)

    merge, drop = summary["merge"], summary["drop"]
    assert merge["n_seeds"] == drop["n_seeds"] == 20
    assert merge["seeds"] == list(range(20))

    f = merge["firing"]
    assert f["prototype_both_positive"] == 20 and f["prototype_beats_classic_both"] == 20
    assert f["wilcoxon_p"]["rho_f"] < 1e-4 and f["wilcoxon_p"]["rho_f_selectivity"] < 1e-4
    assert f["verdict"] is True
    assert drop["firing"]["verdict"] is False
    assert drop["firing"]["wilcoxon_p"]["rho_f_selectivity"] > 0.5  # one-sided, wrong direction

    p = merge["placement"]["inverse"]
    assert p["median_prototype"] == pytest.approx(0.6 + 0.01 * 9.5)
    assert p["ci95"][0] > 0.0 and p["verdict"] is True
    assert drop["placement"]["inverse"]["verdict"] is False

    a = merge["accuracy"]["prototype"]
    assert a["test"]["mean"] == pytest.approx(0.96 + 0.001 * 9.5)
    assert a["test_unfiltered"]["mean"] == pytest.approx(0.94)
    assert merge["accuracy"]["classic"]["test_unfiltered"]["min"] == pytest.approx(0.93 - 0.019)
    assert a["test_calibrated"]["mean"] == pytest.approx(0.97)
    assert a["test_unfiltered_calibrated"]["mean"] == pytest.approx(0.95)

    o = merge["overfitting"]
    assert o["classic"]["seeds_test_kl_rises"] == 20 and o["prototype"]["seeds_test_kl_rises"] == 0
    assert o["classic"]["median_rise"] == pytest.approx(0.03)


def test_calibrated_accuracies_are_absent_unless_computed(tmp_path: Path) -> None:
    for seed in range(3):
        _write_run(tmp_path, seed, "merge", proto_wins=True)
    summary = sw.aggregate(sw.load_runs(sorted(tmp_path.glob("*-seed*-*"))), n_boot=100)
    assert "test_calibrated" not in summary["merge"]["accuracy"]["prototype"]


def test_a_duplicate_seed_or_mixed_commit_is_refused(tmp_path: Path) -> None:
    _write_run(tmp_path / "a", 0, "merge", proto_wins=True)
    _write_run(tmp_path / "b", 0, "merge", proto_wins=True)
    runs = sw.load_runs(sorted(tmp_path.glob("*/*-seed*-*")))
    with pytest.raises(ValueError, match="seed 0"):
        sw.aggregate(runs)

    _write_run(tmp_path / "c", 1, "drop", proto_wins=True, git_hash="abc")
    _write_run(tmp_path / "c", 2, "drop", proto_wins=True, git_hash="def")
    runs = sw.load_runs(sorted((tmp_path / "c").glob("*-seed*-*")))
    with pytest.raises(ValueError, match="git"):
        sw.aggregate(runs)


def test_a_run_without_results_is_reported_not_skipped(tmp_path: Path) -> None:
    d = _write_run(tmp_path, 0, "merge", proto_wins=True)
    (d / "results.json").unlink()
    with pytest.raises(FileNotFoundError, match=r"results\.json"):
        sw.load_runs([d])
