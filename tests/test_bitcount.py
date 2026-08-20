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
