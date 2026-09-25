"""Red-phase tests for the bit-count task [PTQNN Fig 4–6] and the distribution helpers."""

import numpy as np
import pytest

from ptqnn.backends import reference as ref
from ptqnn.circuits import ir, neurons
from ptqnn.tasks import bitcount
from ptqnn.train import distribution, kwsa, losses


def test_target_distribution_matches_fig5_bar_heights() -> None:
    # [PTQNN Fig 5, bottom-right histogram] read-off (±0.001): 00000 → 0.009, one 1-bit → 0.018,
    # two → 0.027, three → 0.036, four → 0.045, 11111 → 0.054, i.e. p(x) = (popcount+1)/112.
    p = bitcount.target_distribution(5)
    assert p.shape == (32,)
    assert p.sum() == pytest.approx(1.0)
    assert p[0b00000] == pytest.approx(0.009, abs=0.001)
    assert p[0b00001] == pytest.approx(0.018, abs=0.001)
    assert p[0b00011] == pytest.approx(0.027, abs=0.001)
    assert p[0b11111] == pytest.approx(0.054, abs=0.001)
    assert p[0b10101] == pytest.approx(p[0b00111])


def test_joint_output_distribution_is_p_x_and_out1_renormalised() -> None:
    # A network whose output is |1⟩ with P(1)=popcount/... : check against explicit marginal.
    net = neurons.Network(n_inputs=2, n_hidden=1, hidden="classic")
    # hidden = classic neuron with weights (π/2, π/2), bias 0 → P(h=1|x) = sin²(popcount·π/4);
    # output CRX(π) from hidden, bias 0 → P(out=1|x) = P(h=1|x).
    params = net.pack(hidden=[((np.pi / 2, np.pi / 2), 0.0)], output=((np.pi,), 0.0))
    prep = [ir.h(0), ir.h(1)]
    q = distribution.conditioned_input_distribution(net, params, prep, ref.StatevectorBackend())
    cond = np.array([np.sin(k * np.pi / 4) ** 2 for k in (0, 1, 1, 2)])
    np.testing.assert_allclose(q, cond / cond.sum(), atol=1e-12)


def test_random_network_kl_is_order_of_fig6_start() -> None:
    # Fig 6: KL at iteration 0 ≈ 0.055 (superposition run). Random prototype nets land in the
    # same decade; this guards the normalisation (a wrong marginal gives KL ≫ 0.1 or ≈ 0).
    net = bitcount.network(n_bits=5, n_hidden=5)
    rng = np.random.default_rng(0)
    kls = [bitcount.kl_loss(net)(net.random_params(rng)) for _ in range(20)]
    assert 0.005 < float(np.median(kls)) < 0.5


@pytest.mark.slow
def test_superposition_training_converges_fast() -> None:
    # Fig 6 (blue): KL → ≈0 within ~8 iterations (from ≈0.055). Seed 0 here: we demand a 5×
    # drop after 4 sweeps and KL < 3e-3 after 20 (the paper's single run looks a little faster;
    # see reports/reproduction.md for the seed spread).
    net = bitcount.network(n_bits=5, n_hidden=5)
    res = bitcount.train_superposition(net, seed=0, iterations=20)
    assert res.history[4] < res.history[0] / 5
    assert res.history[-1] < 3e-3


def test_iterative_scheme_evaluates_single_sample_loss() -> None:
    net = bitcount.network(n_bits=3, n_hidden=2)
    params = net.random_params(np.random.default_rng(1))
    x = 0b101
    loss = bitcount.sample_loss(net, x)(params)
    prep = [ir.x(q) for q in range(3) if (x >> (2 - q)) & 1]
    p1 = distribution.output_activation(net, params, prep, ref.StatevectorBackend())
    assert loss == pytest.approx(losses.squared_error(np.array([p1]), np.array([3 / 4])))


def test_kl_loss_uses_target_first() -> None:
    net = bitcount.network(n_bits=3, n_hidden=1)
    params = net.random_params(np.random.default_rng(2))
    q = distribution.conditioned_input_distribution(
        net, params, bitcount.superposition_prep(net), ref.StatevectorBackend()
    )
    assert bitcount.kl_loss(net)(params) == pytest.approx(
        losses.kl_divergence(bitcount.target_distribution(3), q)
    )
    assert isinstance(kwsa.KWSAConfig(), kwsa.KWSAConfig)


def test_activation_distribution_sums_to_one_and_matches_backend() -> None:
    net = bitcount.network(n_bits=3, n_hidden=2)
    params = net.random_params(np.random.default_rng(0))
    d = bitcount.activation_distribution(net, params)
    assert d.shape == (8,)
    assert d.sum() == pytest.approx(1.0, abs=1e-12)
    be = ref.StatevectorBackend()
    raw = np.array(
        [
            distribution.output_activation(net, params, bitcount.basis_prep(net, x), be)
            for x in range(8)
        ]
    )
    np.testing.assert_allclose(d, raw / raw.sum(), atol=1e-12)


def test_superposition_and_perbasis_distributions_differ() -> None:
    # Claim (a) [session-4 analysis]: the prototype block RX(φ)·CRX·RX(−φ) does not restore the
    # input register, so P(inputs=x, out=1) ≠ 2^-n · P(out=1 | x); the two renormalised
    # distributions differ visibly for random prototype params.
    net = bitcount.network(n_bits=5, n_hidden=5)
    params = net.random_params(np.random.default_rng(0))
    be = ref.StatevectorBackend()
    superposition = distribution.conditioned_input_distribution(
        net, params, bitcount.superposition_prep(net), be
    )
    per_basis = bitcount.activation_distribution(net, params, be)
    assert float(np.max(np.abs(superposition - per_basis))) > 0.01


def test_perbasis_kl_loss_zero_when_model_equals_target(monkeypatch: pytest.MonkeyPatch) -> None:
    net = bitcount.network(n_bits=3, n_hidden=1)
    target = bitcount.target_distribution(3)
    monkeypatch.setattr(
        bitcount, "activation_distribution", lambda net, params, backend=None: target
    )
    assert bitcount.perbasis_kl_loss(net)(np.zeros(net.n_params)) == pytest.approx(0.0, abs=1e-12)


def test_train_iterative_records_both_monitors_on_cadence() -> None:
    # history index i ↔ iteration i·monitor_every; index 0 is the initial point and the final
    # record is always taken, so 64 iterations / 16 → 5 entries in each series.
    net = bitcount.network(n_bits=3, n_hidden=1)
    res = bitcount.train_iterative(net, seed=0, iterations=64, monitor_every=16)
    assert len(res.history) == 5
    assert "superposition" in res.extra_history
    assert len(res.extra_history["superposition"]) == 5

    flipped = bitcount.train_iterative(
        net, seed=0, iterations=64, monitor="superposition", monitor_every=16
    )
    assert "perbasis" in flipped.extra_history
    assert len(flipped.extra_history["perbasis"]) == 5


def test_train_iterative_full_batch_runs_finite() -> None:
    net = bitcount.network(n_bits=5, n_hidden=1)
    res = bitcount.train_iterative(net, seed=0, iterations=4, batch=32)
    assert len(res.history) == 5  # monitor_every defaults to 1
    assert bool(np.all(np.isfinite(res.history)))


def test_train_iterative_anneal_every_defaults_to_dataset_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    net = bitcount.network(n_bits=5, n_hidden=1)
    seen: list[int] = []
    real = kwsa.minimise

    def spy(f, x0, cfg, *args, **kwargs):
        seen.append(kwargs["start_iteration"])
        return real(f, x0, cfg, *args, **kwargs)

    monkeypatch.setattr(bitcount.kwsa, "minimise", spy)
    bitcount.train_iterative(net, seed=0, iterations=33, monitor_every=100)
    assert seen == [0] * 32 + [1]


def test_train_iterative_uses_three_kw_probes_per_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    net = bitcount.network(n_bits=3, n_hidden=1)
    calls = {"n": 0}
    real = bitcount.sample_loss

    def counting(*args, **kwargs):
        loss = real(*args, **kwargs)

        def wrapped(params):
            calls["n"] += 1
            return loss(params)

        return wrapped

    monkeypatch.setattr(bitcount, "sample_loss", counting)
    iterations = 4
    bitcount.train_iterative(net, seed=0, iterations=iterations, monitor_every=1000)
    # One elementary KW step per iteration (n_coords=1, probe_candidates=False): two ±c slope
    # probes + one candidate, plus the fresh sample's incumbent evaluated once.
    assert calls["n"] == 4 * iterations


def test_seed_controls_the_whole_run() -> None:
    net = bitcount.network(n_bits=3, n_hidden=1)
    a = bitcount.train_superposition(net, seed=1, iterations=2)
    b = bitcount.train_superposition(net, seed=1, iterations=2)
    c = bitcount.train_superposition(net, seed=2, iterations=2)
    assert a.history == b.history
    assert a.history != c.history


def test_dead_network_is_penalised() -> None:
    net = bitcount.network(n_bits=2, n_hidden=1)
    assert bitcount.kl_loss(net)(np.zeros(net.n_params)) > 20


def test_dead_perbasis_model_gets_the_epsilon_floor_penalty() -> None:
    # Review 004 M1: a dead network (every P(out=1 | x) = 0) must be penalised through the same
    # eps floor as the superposition KL, not graded as the uniform distribution (KL ≈ 0.065).
    net = bitcount.network(n_bits=2, n_hidden=1)
    zeros = np.zeros(net.n_params)
    np.testing.assert_array_equal(bitcount.activation_distribution(net, zeros), np.zeros(4))
    dead_perbasis = bitcount.perbasis_kl_loss(net)(zeros)
    assert dead_perbasis > 20
    assert dead_perbasis == pytest.approx(bitcount.kl_loss(net)(zeros), rel=1e-9)


def test_monitor_x_places_the_off_cadence_final_point_at_the_last_iteration() -> None:
    # Review 004 M2: 5 iterations monitored every 2 record iterations 0, 2, 4 and the final 5.
    np.testing.assert_array_equal(bitcount.monitor_x(4, 2, 5), [0, 2, 4, 5])
    np.testing.assert_array_equal(bitcount.monitor_x(101, 1000, 100_000), np.arange(101) * 1000)
    np.testing.assert_array_equal(bitcount.monitor_x(4, 1, 3), [0, 1, 2, 3])
    with pytest.raises(ValueError):
        bitcount.monitor_x(3, 2, 5)  # inconsistent with the cadence of train_iterative


def test_monitor_x_matches_the_train_iterative_cadence() -> None:
    net = bitcount.network(n_bits=2, n_hidden=1)
    res = bitcount.train_iterative(net, seed=0, iterations=5, monitor_every=2)
    np.testing.assert_array_equal(bitcount.monitor_x(len(res.history), 2, 5), [0, 2, 4, 5])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"anneal_every": 0}, "anneal_every"),
        ({"anneal_every": -3}, "anneal_every"),
        ({"batch": 0}, "batch"),
        ({"batch": -1}, "batch"),
        ({"batch": 5}, "batch"),  # 2**2 = 4 basis inputs
    ],
)
def test_train_iterative_validates_anneal_every_and_batch(
    kwargs: dict[str, int], match: str
) -> None:
    net = bitcount.network(n_bits=2, n_hidden=1)
    with pytest.raises(ValueError, match=match):
        bitcount.train_iterative(net, seed=0, iterations=1, **kwargs)  # type: ignore[arg-type]
