"""Unit tests for the helpers of ``experiments/flappy_noise.py``.

The script is a driver, but three pieces of it are new code that the whole study rests on:

* :func:`flip_fraction` — how much of the decision map the noise moves (study A/B's headline);
* :func:`binomial_decision` / :func:`flip_probability` — the finite-shot decision rule of
  study C and the ``Φ(−2δ√N)`` prediction of ``reports/hardware-plan.md §1(d)`` it is compared
  against, checked against each other at ≥1000 samples;
* :class:`TransferMatrix` — the exactness claim that makes the study affordable.  It is pinned
  against ``NoisyBackend(DensityMatrixBackend(), model)`` on the **full 128-outcome
  distribution**, not only on ``P(out = 1)`` (``CLAUDE.md``: "test channels, not marginals"),
  and :class:`TablePolicy` is pinned against
  :class:`ptqnn.tasks.flappy.agent.QuantumFlappyPolicy` so the fast path plays the same game.
"""

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from ptqnn.backends import reference as ref
from ptqnn.circuits.neurons import Network
from ptqnn.noise import presets
from ptqnn.noise.model import NoiseModel, NoisyBackend
from ptqnn.tasks.flappy import agent as ag
from ptqnn.train import distribution

_SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "flappy_noise.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("flappy_noise", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["flappy_noise"] = module
    spec.loader.exec_module(module)
    return module


fn = _load()

NET = Network(n_inputs=2, n_hidden=4, hidden="prototype")  # [PTQNN Fig 10]


def _params(seed: int = 3) -> np.ndarray:
    return NET.random_params(np.random.default_rng(seed))


# --- flip fraction ------------------------------------------------------------------------------


def test_flip_fraction_counts_disagreeing_cells() -> None:
    a = np.array([[True, False], [True, True]])
    assert fn.flip_fraction(a, a) == pytest.approx(0.0)
    assert fn.flip_fraction(a, ~a) == pytest.approx(1.0)
    b = a.copy()
    b[0, 0] = False
    assert fn.flip_fraction(a, b) == pytest.approx(0.25)


def test_flip_fraction_weights_by_visits() -> None:
    a = np.array([True, True, False, False])
    b = np.array([False, True, False, False])  # only cell 0 flips
    # Unweighted: 1 of 4.  Weighted by visits that never touch cell 0: nothing flips.
    assert fn.flip_fraction(a, b) == pytest.approx(0.25)
    assert fn.flip_fraction(a, b, [0.0, 10.0, 5.0, 1.0]) == pytest.approx(0.0)
    # Weighted by visits concentrated on cell 0: every played frame flips.
    assert fn.flip_fraction(a, b, [7.0, 0.0, 0.0, 0.0]) == pytest.approx(1.0)
    # A cell visited 3 times out of 12: 3/12.
    assert fn.flip_fraction(a, b, [3.0, 3.0, 3.0, 3.0]) == pytest.approx(0.25)


def test_flip_fraction_no_visits_means_no_flips() -> None:
    a = np.array([True, False])
    assert fn.flip_fraction(a, ~a, [0.0, 0.0]) == pytest.approx(0.0)


def test_flip_fraction_validates() -> None:
    with pytest.raises(ValueError):
        fn.flip_fraction([True, False], [True, False, True])
    with pytest.raises(ValueError):
        fn.flip_fraction([True, False], [True, True], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        fn.flip_fraction([True, False], [True, True], [-1.0, 2.0])
    with pytest.raises(ValueError):
        fn.flip_fraction([True, False], [True, True], [np.nan, 2.0])


# --- finite-shot decision -----------------------------------------------------------------------


def test_binomial_decision_zero_shots_is_the_exact_rule() -> None:
    rng = np.random.default_rng(0)
    # shots <= 0 means "exact expectation value": QuantumFlappyPolicy.act's `p > threshold`.
    assert fn.binomial_decision(0.51, 0, rng) is True
    assert fn.binomial_decision(0.5, 0, rng) is False  # strict `>`, as in agent.act
    assert fn.binomial_decision(0.49, 0, rng) is False


def test_binomial_decision_is_deterministic_at_the_extremes() -> None:
    rng = np.random.default_rng(1)
    for _ in range(20):
        assert fn.binomial_decision(1.0, 100, rng) is True
        assert fn.binomial_decision(0.0, 100, rng) is False


def test_binomial_decision_empirical_rate_matches_the_binomial_tail() -> None:
    # P(decide flap) = P(k > N/2) for k ~ Binomial(N, p); check on 4000 draws at 3 sigma.
    from scipy.stats import binom

    rng = np.random.default_rng(7)
    n_draws, shots, p = 4000, 101, 0.55
    hits = sum(fn.binomial_decision(p, shots, rng) for _ in range(n_draws))
    expected = float(binom.sf(shots // 2, shots, p))  # P(k > 50)
    sd = math.sqrt(expected * (1 - expected) / n_draws)
    assert abs(hits / n_draws - expected) < 3 * sd


def test_binomial_decision_seeded_streams_reproduce() -> None:
    a = [fn.binomial_decision(0.502, 1000, np.random.default_rng([1, 2, 3])) for _ in range(3)]
    b = [fn.binomial_decision(0.502, 1000, np.random.default_rng([1, 2, 3])) for _ in range(3)]
    assert a == b


def test_binomial_decision_rejects_non_probabilities() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        fn.binomial_decision(1.5, 100, rng)
    with pytest.raises(ValueError):
        fn.binomial_decision(-0.2, 100, rng)
    # Floating-point slop from the density-matrix diagonal must not raise.
    assert fn.binomial_decision(1.0 + 1e-12, 10, rng) is True


def test_flip_probability_matches_the_measured_flip_rate() -> None:
    # Φ(−2δ√N) is the normal approximation of P(the sampled decision differs from p > ½).
    rng = np.random.default_rng(11)
    for p, shots in ((0.52, 1000), (0.53, 4000), (0.505, 10000)):
        delta = abs(p - 0.5)
        exact = p > 0.5
        flips = sum(fn.binomial_decision(p, shots, rng) != exact for _ in range(2000))
        predicted = fn.flip_probability(delta, shots)
        assert predicted == pytest.approx(flips / 2000, abs=0.03)


def test_flip_probability_limits() -> None:
    assert fn.flip_probability(0.0, 1000) == pytest.approx(0.5)  # a zero-margin decision is a coin
    assert fn.flip_probability(0.02, 0) == 0.0  # exact expectation never flips
    assert fn.flip_probability(0.5, 10_000) == pytest.approx(0.0, abs=1e-12)
    # Monotone in both arguments.
    assert fn.flip_probability(0.01, 1000) > fn.flip_probability(0.02, 1000)
    assert fn.flip_probability(0.01, 1000) > fn.flip_probability(0.01, 10_000)
    with pytest.raises(ValueError):
        fn.flip_probability(-0.1, 100)


def test_flip_probability_is_the_per_frame_value_not_the_fleet_average() -> None:
    # reports/hardware-plan.md §1(d) reports the *mean* flip probability per frame, 0.35 at
    # N = 100 and 0.077 at N = 4000, from the margin *distribution* of the trained agent.  At
    # small N the curve is flat enough that the mean is the value at the median margin
    # (δ = 0.020); at large N the small-margin tail dominates and the two diverge by an order of
    # magnitude.  This distinction is why the script averages flip_probability over visited
    # frames instead of evaluating it at the median.
    assert fn.flip_probability(0.020, 100) == pytest.approx(0.345, abs=0.01)  # ≈ the table's 0.35
    assert fn.flip_probability(0.020, 4000) == pytest.approx(0.0057, abs=0.0005)  # ≪ 0.077
    # The tail is what makes the mean large: a 1st-percentile margin flips almost half the time.
    assert fn.flip_probability(0.00068, 4000) > 0.4


# --- weighted quantiles -------------------------------------------------------------------------


def test_weighted_quantiles_reduce_to_the_unweighted_ones() -> None:
    v = np.array([0.4, 0.1, 0.3, 0.2])
    got = fn.weighted_quantiles(v, np.ones(4), [0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(got, np.percentile(v, [25, 50, 75, 100], method="inverted_cdf"))


def test_weighted_quantiles_respect_the_weights() -> None:
    # 99 % of the mass sits on 0.001, so every quantile below p99 is 0.001.
    v = np.array([0.001, 0.5])
    w = np.array([99.0, 1.0])
    np.testing.assert_allclose(fn.weighted_quantiles(v, w, [0.1, 0.5, 0.9]), [0.001] * 3)
    assert fn.weighted_quantiles(v, w, [1.0])[0] == pytest.approx(0.5)


def test_weighted_quantiles_validate() -> None:
    with pytest.raises(ValueError):
        fn.weighted_quantiles([1.0, 2.0], [1.0], [0.5])
    with pytest.raises(ValueError):
        fn.weighted_quantiles([1.0, 2.0], [-1.0, 1.0], [0.5])
    with pytest.raises(ValueError):
        fn.weighted_quantiles([1.0, 2.0], [0.0, 0.0], [0.5])


def test_score_stats() -> None:
    s = fn.score_stats([110, 110, 3, 110], perfect=110)
    assert (s["min"], s["max"], s["median"]) == (3.0, 110.0, 110.0)
    assert s["perfect_fraction"] == pytest.approx(0.75)
    assert s["n_episodes"] == 4
    with pytest.raises(ValueError):
        fn.score_stats([])


# --- the transfer matrix: exact against the oracle backend ---------------------------------------


@pytest.mark.parametrize(
    "device",
    [
        None,  # trivial model: no channels at all
        "ibm_fez",  # T1/T2 relaxation + idle scheduling + asymmetric readout
        "quantinuum_h2_1",  # no T1/T2 at all (gate channels + readout only)
        "iqm_garnet",  # zero 1q gate duration, large symmetric readout error
    ],
)
def test_transfer_matrix_reproduces_the_noisy_backend_distribution(device: str | None) -> None:
    # The whole study replaces ~4000 NoisyBackend calls per configuration by 16; the replacement
    # must be exact on the FULL outcome distribution, not just on the P(out=1) marginal.
    params = _params()
    model = fn.noise_model(device)
    backend = NoisyBackend(ref.DensityMatrixBackend(), model)
    tm = fn.TransferMatrix(NET, params, model)
    rng = np.random.default_rng(0)
    for obs in [np.array([0.0, 0.0]), np.array([1.0, 1.0]), *(rng.random(2) for _ in range(3))]:
        prep = ag.encode(obs)
        expected = backend.probabilities(NET.circuit(params, prep))
        np.testing.assert_allclose(tm.probabilities(obs), expected, atol=1e-12)
        exact = distribution.output_activation(NET, params, prep, backend)
        assert tm.activation(obs) == pytest.approx(exact, abs=1e-12)


def test_transfer_matrix_grid_is_the_activation_on_the_quantisation_grid() -> None:
    params = _params(5)
    tm = fn.TransferMatrix(NET, params, fn.noise_model("ibm_fez"))
    levels = 8
    grid = tm.grid(levels)
    assert grid.shape == (levels + 1, levels + 1)
    for i, j in ((0, 0), (3, 7), (levels, levels), (5, 2)):
        assert grid[i, j] == pytest.approx(tm.activation(np.array([i / levels, j / levels])))
    assert np.all(grid >= -1e-12) and np.all(grid <= 1 + 1e-12)


def test_transfer_matrix_noiseless_agrees_with_the_statevector_policy() -> None:
    # With the trivial model the fast noisy path must reproduce the policy the agent was
    # trained with (QuantumFlappyPolicy's fast_linear statevector route).
    params = _params(9)
    tm = fn.TransferMatrix(NET, params, NoiseModel())
    policy = ag.QuantumFlappyPolicy(net=NET, params=params, cache_levels=0)
    rng = np.random.default_rng(2)
    for _ in range(5):
        obs = rng.random(2)
        assert tm.activation(obs) == pytest.approx(policy.activation(obs), abs=1e-12)


def test_transfer_matrix_grid_is_two_feature_only() -> None:
    # The input block is 2^n_inputs-dimensional in general, but the *grid* is the agent's
    # 2-feature quantisation and is refused for anything else.
    other = Network(3, 2, "prototype")
    tm = fn.TransferMatrix(other, other.random_params(np.random.default_rng(0)), NoiseModel())
    assert tm.W.shape == (8, 8)
    with pytest.raises(ValueError):
        tm.grid(4)


# --- the table policy plays the same game as the agent -------------------------------------------


def test_table_policy_matches_quantum_flappy_policy_decisions() -> None:
    params = _params(4)
    levels = 16
    tm = fn.TransferMatrix(NET, params, NoiseModel())
    fast = fn.TablePolicy(tm.grid(levels), levels)
    slow = ag.QuantumFlappyPolicy(net=NET, params=params, cache_levels=levels)
    rng = np.random.default_rng(6)
    for _ in range(200):
        obs = rng.random(2)
        assert fast.act(obs) == slow.act(obs)
    assert fast.frames == 200
    assert fast.visits.sum() == 200


def test_table_policy_records_shot_flips() -> None:
    levels = 4
    table = np.full((levels + 1, levels + 1), 0.5 + 1e-9)  # margin ≈ 0: every frame is a coin
    policy = fn.TablePolicy(table, levels, shots=100, rng=np.random.default_rng(0))
    for _ in range(400):
        policy.act(np.array([0.5, 0.5]))
    assert policy.frames == 400
    assert 0.3 < policy.flips / policy.frames < 0.7  # a zero-margin decision flips ~half the time


def test_table_policy_validates() -> None:
    with pytest.raises(ValueError):
        fn.TablePolicy(np.zeros((3, 3)), levels=4)
    with pytest.raises(ValueError):
        fn.TablePolicy(np.zeros((5, 5)), levels=4, shots=100)  # finite shots need an rng


# --- noise model selection ------------------------------------------------------------------------


def test_noise_model_selection() -> None:
    assert fn.noise_model(None).is_trivial()
    assert fn.noise_model("ibm_fez", 0.0).is_trivial()
    base = presets.device("ibm_fez").noise_model()
    assert fn.noise_model("ibm_fez", 1.0) is base or fn.noise_model("ibm_fez").t1_us == base.t1_us
    assert fn.noise_model("ibm_fez", 2.0).t1_us == pytest.approx(base.t1_us / 2.0)
