"""Unit tests for the pure helpers of ``experiments/ecg_iris_noise.py``.

The script owns no physics — it drives ``ptqnn.noise``, ``ptqnn.tasks`` and ``ptqnn.backends`` —
but it does own four small estimators that are new code: the evaluation-block chooser, the
density-matrix cost model that turns a timing probe into a window budget, the binomial shot
sampler with its closed-form shot requirement, and the single-call Iris read-out.  Each is
tested here against a value known by hand or against the library function it stands in for.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "ecg_iris_noise.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ecg_iris_noise", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["ecg_iris_noise"] = module
    spec.loader.exec_module(module)
    return module


ein = _load()


# --- argument parsing ---------------------------------------------------------------------------


def test_list_parsers_accept_commas_and_spaces() -> None:
    assert ein.parse_list("a,b c") == ["a", "b", "c"]
    assert ein.parse_ints("0,1000, 10000") == [0, 1000, 10000]
    assert ein.parse_floats("0,0.5,1,2,4") == [0.0, 0.5, 1.0, 2.0, 4.0]
    assert ein.parse_list("") == []


# --- evaluation block ---------------------------------------------------------------------------


def test_centred_block_is_contiguous_and_centred() -> None:
    idx = np.arange(100, 200)
    b = ein.centred_block(idx, 10)
    assert b.tolist() == list(range(145, 155))
    assert np.all(np.diff(b) == 1)


def test_centred_block_returns_everything_when_not_reducing() -> None:
    idx = np.arange(5, 11)
    assert ein.centred_block(idx, 0).tolist() == idx.tolist()
    assert ein.centred_block(idx, 99).tolist() == idx.tolist()


# --- cost model ---------------------------------------------------------------------------------


def test_cost_units_counts_one_reference_evaluation_per_extra_qubit() -> None:
    # A cell with k hidden neurons runs on 5 + k + 1 qubits; a density-matrix step is O(4^n), so
    # relative to the k = 3 cell the k = 2 cell costs 1/4 and the k = 1 cell 1/16.
    assert ein.cost_units([3]) == pytest.approx(1.0)
    assert ein.cost_units([1, 2, 3]) == pytest.approx(1 / 16 + 1 / 4 + 1)
    assert ein.cost_units([1, 2, 3, 1, 2, 3]) == pytest.approx(2 * (1 / 16 + 1 / 4 + 1))


def test_budget_windows_inverts_the_cost_model() -> None:
    # 100 s at 0.1 s per reference evaluation, 5 combos, 2 units per window: 100 / (0.1·5·2) = 100.
    assert ein.budget_windows(100.0, 0.1, 5, 2.0) == 100
    assert ein.budget_windows(1e-9, 0.1, 5, 2.0) == 1  # never zero windows
    with pytest.raises(ValueError):
        ein.budget_windows(100.0, 0.0, 5, 2.0)


# --- shot noise -----------------------------------------------------------------------------------


def test_shot_activations_are_unbiased_with_binomial_variance() -> None:
    rng = np.random.default_rng(0)
    p = np.array([0.1, 0.5, 0.9])
    shots = 200
    draws = np.array([ein.shot_activations(p, shots, rng) for _ in range(4000)])
    assert np.allclose(draws.mean(axis=0), p, atol=4 * np.sqrt(p * (1 - p) / shots / 4000) + 1e-3)
    assert np.allclose(draws.var(axis=0), p * (1 - p) / shots, rtol=0.1)
    assert np.all((draws >= 0) & (draws <= 1))


def test_shot_activations_reject_zero_shots() -> None:
    with pytest.raises(ValueError):
        ein.shot_activations([0.5], 0, np.random.default_rng(0))


def test_shots_for_db_loss_matches_the_psnr_it_is_derived_from() -> None:
    mse, var, db = 1.2e-4, 0.24, 1.0
    n = ein.shots_for_db_loss(mse, var, db)
    # By construction the PSNR at exactly n shots is db below the infinite-shot value.
    r = 0.81
    psnr_inf = 10 * np.log10(r**2 / mse)
    psnr_n = 10 * np.log10(r**2 / (mse + var / n))
    assert psnr_inf - psnr_n == pytest.approx(db)


def test_shot_study_agrees_with_its_own_closed_form() -> None:
    rng = np.random.default_rng(1)
    y = rng.uniform(0.2, 0.8, 400)
    p = np.clip(y + rng.normal(0, 0.01, y.size), 0, 1)
    out = ein.shot_study(p, y, float(np.ptp(y)), [0, 2000], 12, 7)
    row = out["rows"][1]
    assert row["shots"] == 2000
    # 12 repetitions: the sample mean must sit within ~4 standard errors of the prediction.
    assert row["psnr_mean"] == pytest.approx(
        row["psnr_predicted"], abs=4 * row["psnr_sd"] / np.sqrt(12) + 0.05
    )
    assert out["rows"][0]["shots"] == 0
    assert out["rows"][0]["psnr_mean"] == pytest.approx(out["exact_psnr"])


# --- reporting helpers ----------------------------------------------------------------------------


def test_devices_above_sorts_best_first_and_filters() -> None:
    rows = {"a": 34.0, "b": 39.0, "c": 36.0}
    assert ein.devices_above(rows, 35.0) == ["b", "c"]
    assert ein.devices_above(rows, 40.0) == []


def test_loglog_exponent_recovers_a_known_power_law() -> None:
    s = [0.0, 0.5, 1.0, 2.0, 4.0]
    floor = 3.0
    v = [floor + 0.7 * x**1.8 for x in s]
    assert ein.loglog_exponent(s, v, floor) == pytest.approx(1.8, abs=1e-6)
    # A noise model that *improves* the metric leaves no positive excess: report nan, not a fit.
    assert np.isnan(ein.loglog_exponent(s, [floor - 0.1 * x for x in s], floor))


def test_shot_cell_formats_missing_budgets() -> None:
    assert ein._shot_cell(None) == "—"
    assert ein._shot_cell({"psnr_mean": 31.234, "psnr_sd": 0.5}) == "31.23±0.50"


# --- Iris single-call read-out --------------------------------------------------------------------


def test_iris_scores_reproduces_the_library_kl_and_accuracy() -> None:
    """The script derives KL *and* accuracy from one backend call; the library uses two.

    ``model.kl_loss`` renormalises ``P(inputs = x, out = 1)`` over all 4096 input strings, while
    ``model.accuracy`` post-selects onto the dataset codes first.  Both come from the same joint
    vector, so a single backend call suffices — this pins that equivalence at 17 qubits, where a
    doubled trajectory budget is the difference between feasible and not.
    """
    from ptqnn.backends import reference as ref
    from ptqnn.tasks.iris import data as iris_data
    from ptqnn.tasks.iris import model as iris_model
    from ptqnn.tasks.iris import train as iris_train

    split = iris_data.load_split(seed=0, train_size=1.0 / 3.0)
    cfg = iris_train.IrisConfig(iterations=1, n_hidden=2)
    net = iris_train.network(cfg, "prototype")
    params = net.random_params(np.random.default_rng(3))
    codes, labels = split.test_codes, split.test_labels
    psi0 = iris_model.dataset_superposition(codes, net.n_qubits, net.input_qubits)

    p = ref.StatevectorBackend().probabilities(net.circuit(params, initial_state=psi0))
    joint = ref.marginal(p, [*net.input_qubits, net.output_qubit]).reshape(2**net.n_inputs, 2)[:, 1]
    got = ein.iris_scores(joint, codes, labels, iris_model.target_distribution(codes, labels))

    assert got["kl"] == pytest.approx(
        iris_model.kl_loss(net, codes, labels, ref.StatevectorBackend())(params)
    )
    assert got["accuracy"] == pytest.approx(
        iris_model.accuracy(net, params, codes, labels, ref.StatevectorBackend())
    )
    assert got["leak"] == pytest.approx(0.0, abs=1e-9) or 0.0 <= got["leak"] <= 1.0


def test_iris_scores_handles_a_network_that_never_fires() -> None:
    joint = np.zeros(2**4)
    codes = np.array([0, 3, 5], dtype=np.int64)
    labels = np.array([0, 1, 2], dtype=np.int64)
    target = np.zeros(2**4)
    target[codes] = np.array([1.0, 4.0, 9.0]) / 14.0
    got = ein.iris_scores(joint, codes, labels, target)
    assert np.isfinite(got["kl"]) and got["kl"] > 1.0
    assert 0.0 <= got["accuracy"] <= 1.0


# --- ECG metrics wrapper --------------------------------------------------------------------------


def test_ecg_metrics_are_the_paper_equations() -> None:
    from ptqnn.tasks.ecg import model as ecg_model

    rng = np.random.default_rng(5)
    y = rng.uniform(0.2, 0.8, 200)
    yhat = y + rng.normal(0, 0.01, y.size)
    m = ein.ecg_metrics(y, yhat, y, 0.8)
    assert m["psnr"] == pytest.approx(ecg_model.psnr(y, yhat, 0.8))
    assert m["ssim_windowed"] == pytest.approx(ecg_model.ssim_windowed(y, yhat))
    assert m["mse"] == pytest.approx(ecg_model.mse(y, yhat))
    # Scored against a perfect reference, "vs clean model" and "vs truth" coincide.
    assert m["psnr_vs_clean_model"] == pytest.approx(m["psnr"])
