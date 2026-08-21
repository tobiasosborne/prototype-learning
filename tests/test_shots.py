"""Shot-noise programme items 2–3: a sampling backend with common random numbers for KW–SA
probe pairs, and a sequential (SPRT) decision rule for the Flappy policy."""

from __future__ import annotations

import numpy as np
from scipy.stats import binom

from ptqnn.backends import reference as ref
from ptqnn.backends.shots import ShotBackend
from ptqnn.circuits import ir
from ptqnn.circuits.neurons import Network
from ptqnn.tasks import bitcount
from ptqnn.tasks.flappy import agent as ag
from ptqnn.train import kwsa


def _circ() -> ir.Circuit:
    return ir.Circuit(2, ops=[ir.h(0), ir.rx(1, 0.7), ir.crx(0, 1, 1.1)])


def test_shot_backend_is_a_multinomial_estimate_of_the_inner_distribution() -> None:
    exact = ref.StatevectorBackend().probabilities(_circ())
    be = ShotBackend(ref.StatevectorBackend(), shots=4000, seed=0)
    ps = np.array([be.probabilities(_circ()) for _ in range(50)])
    assert np.allclose(ps.sum(axis=1), 1.0)
    assert be.shots_used == 50 * 4000
    # 50 × 4000 draws: mean within 4 σ of the exact value on every outcome
    mean = ps.mean(axis=0)
    sd = np.sqrt(exact * (1 - exact) / (50 * 4000))
    assert np.all(np.abs(mean - exact) <= 4 * sd + 1e-9)
    assert "shots=4000" in be.name


def test_shot_backend_draws_differ_between_calls_but_repeat_inside_a_pair() -> None:
    be = ShotBackend(ref.StatevectorBackend(), shots=100, seed=3)
    a = be.probabilities(_circ())
    b = be.probabilities(_circ())
    assert not np.array_equal(a, b)
    with be.common_random_numbers():
        c = be.probabilities(_circ())
        d = be.probabilities(_circ())
    assert np.array_equal(c, d)  # same circuit, same draws → identical estimate
    with be.common_random_numbers():
        e = be.probabilities(_circ())
    assert not np.array_equal(c, e)  # the next pair gets fresh common randomness


def test_kwsa_probe_context_wraps_each_coordinate_triple() -> None:
    entered: list[int] = []
    calls: list[int] = []

    class Ctx:
        def __enter__(self) -> None:
            entered.append(len(calls))

        def __exit__(self, *a: object) -> None:
            pass

    def f(x: np.ndarray) -> float:
        calls.append(1)
        return float(np.sum(x**2))

    cfg = kwsa.KWSAConfig(iterations=2, a0=0.1, c0=0.1, seed=0)
    kwsa.minimise(f, np.ones(3), cfg, probe_context=Ctx)
    # 1 initial evaluation, then per iteration 3 coordinates × (one context of 3 probes)
    assert len(entered) == 2 * 3
    assert entered == [1, 4, 7, 10, 13, 16]


def test_common_random_numbers_cut_finite_difference_variance() -> None:
    """The point of item 2: with maximally coupled probes the central difference only sees the
    ``N·TV(p₊, p₋) ∝ N·c`` shots that change outcome, so its variance is ``O(c/N)`` instead of
    the independent ``O(1/(c²N))`` — the ratio grows like ``1/c``.  Measured on the bit-count KL
    at one coordinate (1000 gradient samples each): 2.9× at c = 0.2, 12–22× at c = 0.05.  The
    test pins the small-step regime, where finite differences are otherwise unusable."""
    net = bitcount.network(3, 2)
    rng = np.random.default_rng(0)
    x = net.random_params(rng)
    c = 0.05
    e = np.zeros_like(x)
    e[0] = c

    def grads(crn: bool) -> np.ndarray:
        be = ShotBackend(ref.StatevectorBackend(), shots=400, seed=1)
        loss = bitcount.kl_loss(net, be)
        out = []
        for _ in range(1000):
            if crn:
                with be.common_random_numbers():
                    fp, fm = loss(x + e), loss(x - e)
            else:
                fp, fm = loss(x + e), loss(x - e)
            out.append((fp - fm) / (2 * c))
        return np.asarray(out)

    g_crn, g_ind = grads(True), grads(False)
    assert g_ind.var() > 5 * g_crn.var(), (g_ind.var(), g_crn.var())
    # both are unbiased for the same (exact) derivative
    exact = (bitcount.kl_loss(net)(x + e) - bitcount.kl_loss(net)(x - e)) / (2 * c)
    assert abs(g_crn.mean() - exact) < 4 * g_crn.std() / np.sqrt(1000)


# --- SPRT decisions -------------------------------------------------------------------------------


def test_sprt_stops_early_far_from_the_boundary_and_is_slow_near_it() -> None:
    rule = ag.SPRTDecision(delta0=0.02, alpha=0.05, beta=0.05, max_shots=10_000, block=50)
    rng = np.random.default_rng(0)
    far = [rule.decide(0.9, rng) for _ in range(200)]
    assert all(d for d, _ in far)
    far_mean = np.mean([n for _, n in far])
    assert far_mean < 200
    # at p = ½ the log-likelihood ratio is a symmetric random walk: it still terminates, but
    # only after ~(boundary/step)² ≈ 1 300 shots on average, and never beyond the cap.
    near = [n for _, n in (rule.decide(0.5, rng) for _ in range(200))]
    assert max(near) <= 10_000
    assert np.mean(near) > 10 * far_mean


def test_sprt_error_rate_at_the_indifference_margin_is_bounded() -> None:
    """At ``p = ½ + δ0`` Wald's bound gives error ≤ β (one-sided); 2000 trials, 3 σ slack."""
    rule = ag.SPRTDecision(delta0=0.03, alpha=0.05, beta=0.05, max_shots=20_000, block=20)
    rng = np.random.default_rng(1)
    wrong = np.mean([not rule.decide(0.53, rng)[0] for _ in range(2000)])
    assert wrong < 0.05 + 3 * np.sqrt(0.05 * 0.95 / 2000)
    # and far cheaper than the fixed-N budget that reaches the same error at this margin
    n_fixed = int(np.ceil((1.645 / (2 * 0.03)) ** 2))  # Φ(−2δ√N) = 0.05
    mean_shots = np.mean([rule.decide(0.53, rng)[1] for _ in range(200)])
    assert mean_shots < n_fixed
    assert binom  # scipy import used for clarity of the binomial model in the docstrings


def test_policy_with_sprt_rule_uses_fewer_shots_than_fixed_n() -> None:
    net = Network(n_inputs=2, n_hidden=2, hidden="prototype")
    params = net.random_params(np.random.default_rng(7))
    from ptqnn.tasks.flappy import train as tr

    fixed = tr.make_policy(net, params, tr.ObjectiveConfig(), shots=4000, shot_seed=0)
    sprt = tr.make_policy(
        net,
        params,
        tr.ObjectiveConfig(),
        shots=4000,
        shot_seed=0,
        decision=ag.SPRTDecision(delta0=0.02, max_shots=4000),
    )
    for obs in np.random.default_rng(1).uniform(size=(30, 2)):
        fixed.act(obs)
        sprt.act(obs)
    assert fixed.shots_used == 30 * 4000
    assert 0 < sprt.shots_used < fixed.shots_used
