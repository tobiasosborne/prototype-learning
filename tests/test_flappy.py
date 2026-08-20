"""Flappy Bird ``[PTQNN §III.D, Fig 10, 11]``: clone, encoding, policy, training."""

import numpy as np
import pytest

from ptqnn.backends import reference as ref
from ptqnn.circuits.neurons import Network
from ptqnn.tasks.flappy import agent as ag
from ptqnn.tasks.flappy import env as fe
from ptqnn.tasks.flappy import train as tr
from ptqnn.train import distribution, kwsa

# --- environment ----------------------------------------------------------------------------


def _play(actions, seed=0, config=None):
    e = fe.FlappyEnv(config=config or fe.FlappyConfig(), seed=seed)
    obs = [e.reset()]
    for a in actions:
        if e.done:
            break
        obs.append(e.step(a)[0])
    return e, np.array(obs)


def test_env_is_deterministic_under_a_seed() -> None:
    rng = np.random.default_rng(3)
    actions = [bool(b) for b in rng.integers(0, 2, 200)]
    a, obs_a = _play(actions, seed=7)
    b, obs_b = _play(actions, seed=7)
    np.testing.assert_array_equal(obs_a, obs_b)
    assert (a.score, a.steps, a.done) == (b.score, b.steps, b.done)
    assert [p.gap_centre for p in a.state.pipes] == [p.gap_centre for p in b.state.pipes]


def test_different_seeds_give_different_gap_heights() -> None:
    a = fe.FlappyEnv(seed=1).state.pipes
    b = fe.FlappyEnv(seed=2).state.pipes
    assert [p.gap_centre for p in a] != [p.gap_centre for p in b]
    lo, hi = fe.FlappyConfig().gap_centre_range
    assert all(lo <= p.gap_centre <= hi for p in a + b)


def test_reset_restores_the_initial_state() -> None:
    e = fe.FlappyEnv(seed=4)
    start = e.observation()
    e.run([False] * 5)
    assert not np.allclose(e.observation(), start)
    np.testing.assert_allclose(e.reset(), start)
    assert (e.score, e.steps, e.done) == (0, 0, False)


def test_observation_is_pure_and_normalised() -> None:
    e = fe.FlappyEnv(seed=0)
    e.run([True, False, False])
    before = e.observation()
    for _ in range(3):
        np.testing.assert_array_equal(e.observation(), before)
    assert before.shape == (2,)
    assert np.all((before >= 0.0) & (before <= 1.0))
    cfg = e.config
    assert before[0] == pytest.approx(e.state.bird_y / cfg.playable_height)
    nxt = fe.next_pipe(cfg, e.state)
    assert nxt is not None
    assert before[1] == pytest.approx(nxt.gap_centre / cfg.playable_height)


def test_feature_sets_are_pluggable() -> None:
    assert fe.n_features("bird_gap") == 2
    assert fe.n_features("bird_gap_distance") == 3
    e = fe.FlappyEnv(seed=0, features="bird_gap_distance")
    obs = e.observation()
    assert obs.shape == (3,)
    np.testing.assert_allclose(obs[:2], fe.FlappyEnv(seed=0).observation())
    # the extra feature is the horizontal distance and shrinks as the pipe scrolls in
    assert obs[2] == pytest.approx((288.0 - 40.0) / 288.0)
    e.run([False, False, False])
    assert e.observation()[2] < obs[2]


def test_bird_falls_and_dies_without_flaps() -> None:
    # From mid-screen (200 px) the bird must fall 200 − 12 px; with g = 1 px/frame² that is
    # t(t+1)/2 ≥ 188 → t ≈ 19 frames.
    e, _ = _play([False] * 200, seed=0)
    assert e.done
    assert 15 <= e.steps <= 25
    assert e.state.bird_y + 0.5 * e.config.bird_height >= e.config.playable_height
    assert e.score == 0


def test_holding_the_flap_hits_the_ceiling() -> None:
    e, _ = _play([True] * 200, seed=0)
    assert e.done
    assert e.state.bird_y - 0.5 * e.config.bird_height <= 0.0
    assert 20 <= e.steps <= 35


def test_stepping_a_finished_episode_raises() -> None:
    e, _ = _play([False] * 200, seed=0)
    with pytest.raises(RuntimeError):
        e.step(False)


def test_terminal_velocity_is_capped() -> None:
    cfg = fe.FlappyConfig(start_height_fraction=0.05, gap_margin=80.0)
    e = fe.FlappyEnv(config=cfg, seed=0)
    speeds = []
    for _ in range(30):
        if e.done:
            break
        e.step(False)
        speeds.append(e.state.bird_vy)
    assert max(speeds) <= cfg.max_fall_speed + 1e-12
    assert max(speeds) == pytest.approx(cfg.max_fall_speed)


def test_classical_baseline_passes_fifty_pipes() -> None:
    # Proof that the two paper features suffice: flap iff the bird is below the gap centre.
    # 3000 frames at 45 frames/pipe is 66 pipes if the bird never dies.
    for seed in (0, 1, 2):
        e = fe.FlappyEnv(seed=seed)
        res = ag.play_episode(e, tr.baseline_policy(), max_steps=3000)
        assert res.survived, f"baseline died after {res.steps} frames on seed {seed}"
        assert res.score >= 50, f"seed {seed}: only {res.score} pipes"


def test_reward_counts_frames_and_pipes() -> None:
    cfg = fe.FlappyConfig()
    e = fe.FlappyEnv(config=cfg, seed=0)
    res = ag.play_episode(e, tr.baseline_policy(), max_steps=500)
    assert res.total_reward == pytest.approx(
        res.steps * cfg.step_reward + res.score * cfg.pipe_reward
    )


# --- encoding -------------------------------------------------------------------------------


def test_encode_maps_features_to_rx_angles() -> None:
    # Default map θ = 2·asin(√x) [SNN Fig 4, §III.A], pinned by that figure's printed
    # CRX(1.369) = 2·asin(√0.4) and CRX(1.7722) = 2·asin(√0.6).
    ops = ag.encode(np.array([0.0, 0.4, 0.6, 1.0]))
    assert [op.name for op in ops] == ["rx"] * 4
    assert [op.qubits for op in ops] == [(0,), (1,), (2,), (3,)]
    assert [op.params[0] for op in ops] == pytest.approx([0.0, 1.369, 1.7722, np.pi], abs=5e-4)


def test_encoded_angle_carries_the_feature_as_a_probability() -> None:
    # The point of the asin map: P(qubit = 1) = sin²(θ/2) = x exactly [SNN §III.A].
    for x in (0.0, 0.13, 0.5, 0.77, 1.0):
        theta = ag.encode([x])[0].params[0]
        assert np.sin(theta / 2) ** 2 == pytest.approx(x)


def test_linear_angle_map_is_available_for_ablation() -> None:
    # θ = π·x, the naive reading of the ECG figure [PTQNN Fig 12]; agrees with the default
    # at x = 0, 0.5 and 1 and differs in between.
    ops = ag.encode(np.array([0.0, 0.25, 0.5, 1.0]), mapping="linear")
    assert [op.params[0] for op in ops] == pytest.approx([0.0, np.pi / 4, np.pi / 2, np.pi])
    assert ag.encode([0.5])[0].params[0] == pytest.approx(np.pi / 2)
    assert ag.encode([0.25])[0].params[0] != pytest.approx(np.pi / 4)


def test_encode_scale_is_a_parameter() -> None:
    ops = ag.encode([0.5], mapping="linear", scale=2.0)
    assert ops[0].params[0] == pytest.approx(2.0 * np.pi * 0.5)
    assert ag.encode([1.0], scale=0.5)[0].params[0] == pytest.approx(np.pi / 2)


def test_fig10_input_angles_decode_to_plausible_features() -> None:
    # [PTQNN Fig 10] shows RX(1.9537) and RX(2.1962) on the two input qubits.  Under the pinned
    # map those are the features sin²(θ/2) = 0.686 and 0.792 — a bird below mid-screen with the
    # gap below it; which feature sits on which qubit is [ASSUMPTION].
    x = [float(np.sin(t / 2) ** 2) for t in (1.9537, 2.1962)]
    assert x == pytest.approx([0.6863, 0.7923], abs=1e-3)
    ops = ag.encode(x)
    assert [op.params[0] for op in ops] == pytest.approx([1.9537, 2.1962], abs=1e-3)


# --- policy ---------------------------------------------------------------------------------


def _net() -> Network:
    # [PTQNN Fig 10]: 2 inputs, 4 prototype hidden neurons (qubits 3–6), 1 output (qubit 7).
    return Network(n_inputs=2, n_hidden=4, hidden="prototype")


def test_network_matches_the_fig10_circuit() -> None:
    net = _net()
    assert net.n_qubits == 7
    assert net.hidden_qubits == (2, 3, 4, 5) and net.output_qubit == 6
    params = net.pack(
        hidden=[((0.1, 0.2), 0.3)] * 4,
        output=((-1.0422, -0.9988, -1.2622, -0.91002), -1.4277),
    )
    circ = net.circuit(params, prep=ag.encode([1.9537 / np.pi, 2.1962 / np.pi]))
    assert circ.count("crx") == 4 * 2 + 4  # 2 per prototype neuron + 4 into the output
    assert [op.params[0] for op in circ.ops if op.name == "rx"][-1] == pytest.approx(-1.4277)


def test_fast_linear_path_agrees_with_the_reference_activation() -> None:
    # The fast path evaluates U_tail on the 4-dimensional input basis and superposes; it must
    # be bit-for-bit the same physics as ptqnn.train.distribution.output_activation.
    net = _net()
    rng = np.random.default_rng(0)
    params = net.random_params(rng)
    fast = ag.QuantumFlappyPolicy(net, params, cache_levels=0)
    for _ in range(10):
        obs = rng.uniform(0.0, 1.0, 2)
        oracle = distribution.output_activation(
            net, params, ag.encode(obs), ref.StatevectorBackend()
        )
        assert fast.activation(obs) == pytest.approx(oracle, abs=1e-12)
        slow = ag.QuantumFlappyPolicy(net, params, cache_levels=0, fast_linear=False)
        assert slow.activation(obs) == pytest.approx(oracle, abs=1e-12)


def test_policy_flaps_iff_activation_exceeds_threshold() -> None:
    net = _net()
    params = net.random_params(np.random.default_rng(1))
    obs = np.array([0.25, 0.75])  # on the 1/64 cache grid, so quantisation is a no-op
    p = ag.QuantumFlappyPolicy(net, params).activation(obs)
    assert 0.0 < p < 1.0
    assert ag.QuantumFlappyPolicy(net, params, threshold=p - 1e-9).act(obs) is True
    assert ag.QuantumFlappyPolicy(net, params, threshold=p + 1e-9).act(obs) is False


def test_decision_cache_is_keyed_on_quantised_observations() -> None:
    net = _net()
    policy = ag.QuantumFlappyPolicy(net, net.random_params(np.random.default_rng(2)))
    a = policy.act(np.array([0.5, 0.5]))
    assert policy.decision_cache == {(32, 32): a}
    assert policy.act(np.array([0.5001, 0.4999])) is a  # same cell, no new entry
    assert len(policy.decision_cache) == 1
    policy.act(np.array([0.9, 0.1]))
    assert len(policy.decision_cache) == 2
    # a long game must not re-simulate: the cache is bounded by the grid, not by the frames
    ag.play_episode(fe.FlappyEnv(seed=0), policy, max_steps=2000)
    assert len(policy.decision_cache) <= 65 * 65


def test_threshold_policy_flaps_when_below_the_gap() -> None:
    p = tr.baseline_policy()
    assert p.act(np.array([0.7, 0.5])) is True  # bird below the gap (y grows downwards)
    assert p.act(np.array([0.3, 0.5])) is False


# --- rendering ------------------------------------------------------------------------------


def test_render_frame_is_an_rgb_raster() -> None:
    e = fe.FlappyEnv(seed=0)
    e.run([True] * 10)
    img = ag.render_frame(e)
    cfg = e.config
    assert img.shape == (int(cfg.screen_height), int(cfg.screen_width), 3)
    assert img.dtype == np.uint8
    assert img[-1, 0].tolist() == list(ag._GROUND)
    # the bird is drawn where the state says it is
    y, x = int(e.state.bird_y), int(cfg.bird_x)
    assert img[y, x - 12].tolist() == list(ag._BIRD)


def test_render_episode_returns_one_frame_per_step() -> None:
    res = ag.play_episode(fe.FlappyEnv(seed=0), tr.baseline_policy(), max_steps=20, record=True)
    frames = ag.render_episode(fe.FlappyConfig(), res.trajectory, max_frames=5)
    assert len(frames) == 5
    assert all(f.shape == (512, 288, 3) for f in frames)


# --- training -------------------------------------------------------------------------------


def test_objective_is_negative_shaped_return() -> None:
    net = _net()
    cfg = tr.ObjectiveConfig(max_steps=200)
    params = net.random_params(np.random.default_rng(0))
    loss = tr.objective(net, seeds=(0, 1), cfg=cfg)(params)
    policy = tr.make_policy(net, params, cfg)
    results = tr.evaluate(policy, (0, 1), cfg.max_steps, cfg.env_config)
    assert loss == pytest.approx(tr.episode_loss(results, cfg.shaping_weight))
    assert loss > -1.0  # a random 17-parameter network does not survive


def test_objective_prefers_the_baseline_behaviour() -> None:
    # sanity check on the shaping: the classical solution must score far better than random.
    net = _net()
    cfg = tr.ObjectiveConfig(max_steps=400)
    good = tr.episode_loss(tr.evaluate(tr.baseline_policy(), (0, 1), 400), cfg.shaping_weight)
    rng = np.random.default_rng(0)
    random_losses = [
        tr.objective(net, seeds=(0, 1), cfg=cfg)(net.random_params(rng)) for _ in range(5)
    ]
    assert good < min(random_losses) - 1.0


@pytest.mark.slow
def test_quantum_policy_learns_to_pass_twenty_pipes() -> None:
    # Bounded budget: 200 random restarts on 250-frame episodes + 12 KW-SA coordinate sweeps on
    # 800-frame episodes over three seeded episodes (~20 s wall clock, ~2000 loss evaluations).
    # We do not attempt the paper's 6-day / 450 000-pipe run [PTQNN §III.D]; the target is
    # deterministic play on held-out seeds under a 5000-frame cap (110 pipes = perfect).
    net = _net()
    cfg = kwsa.KWSAConfig(**{**tr.DEFAULT_KWSA.__dict__, "iterations": 12, "seed": 0})
    res = tr.train(
        net,
        cfg=cfg,
        seeds=(0, 1, 2),
        max_steps=800,
        n_restarts=200,
        eval_seeds=(5, 6, 7),
        eval_steps=5000,
    )
    assert res.loss < res.init_loss
    assert min(res.scores) >= 20, f"scores {res.scores} (loss {res.loss:.3f})"
