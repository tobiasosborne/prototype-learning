"""Red-phase tests for the Iris task ``[PTQNN §III.C, Fig 7–9]``, ``[SNN §IV.B]``."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from ptqnn.backends import reference as ref
from ptqnn.circuits.neurons import Network
from ptqnn.tasks.iris import data as iris_data
from ptqnn.tasks.iris import model as iris_model
from ptqnn.tasks.iris import train as iris_train
from ptqnn.train import losses


@pytest.fixture(scope="module")
def split() -> iris_data.IrisSplit:
    return iris_data.load_split(seed=0)


def test_every_code_has_exactly_four_ones(split: iris_data.IrisSplit) -> None:
    # [SNN §IV.B] "every datapoint can be encoded in a 4 × 3 = 12−dimensional binary vector
    # which contains exactly 4 non-zero entries."
    for codes in (split.train_codes, split.test_codes):
        assert codes.min() >= 0 and codes.max() < 2**12
        popcount = np.array([bin(int(c)).count("1") for c in codes])
        np.testing.assert_array_equal(popcount, 4)
        # exactly one bit inside each 3-bit feature block
        bits = iris_data.code_bits(codes)
        np.testing.assert_array_equal(bits.reshape(len(codes), 4, 3).sum(axis=2), 1)


def test_centroids_are_fitted_on_the_training_split_only(split: iris_data.IrisSplit) -> None:
    # [SNN §IV.B] "After separating training and test data, a kMeans clustering on each
    # dimension with k = 3 is used on the training data. ... For the given cluster centers,
    # the same can be done with the test data."
    x, y = iris_data.load_iris_xy()
    tr = split.train_index
    np.testing.assert_allclose(split.centers, iris_data.fit_centers(x[tr], seed=split.seed))
    # the centres are *not* those of the full dataset (else the fit leaked the test data)
    assert not np.allclose(split.centers, iris_data.fit_centers(x, seed=split.seed))
    # and the test codes come from applying those very centres
    np.testing.assert_array_equal(
        split.test_codes, iris_data.encode_dataset(x[split.test_index], split.centers)[0]
    )
    assert y.shape == (150,)


def test_collisions_are_counted_and_merged_by_average_label(split: iris_data.IrisSplit) -> None:
    codes = np.array([5, 5, 6], dtype=np.int64)
    labels = np.array([0, 2, 1], dtype=np.int64)
    merged = iris_data.merge_duplicate_codes(codes, labels)
    np.testing.assert_array_equal(merged.codes, [5, 6])
    np.testing.assert_allclose(merged.labels, [1.0, 1.0])  # [ASSUMPTION] average label
    assert merged.n_collisions == 1
    assert (
        split.train_collisions
        == iris_data.merge_duplicate_codes(split.train_codes, split.train_labels).n_collisions
    )


def test_drop_ambiguous_codes_removes_codes_with_two_labels() -> None:
    # [Rosenhahn e-mail 2026-08-21] "On the iris-dataset I should have mentioned that I removed
    # binary patterns which were assigned to two classes."  Code 5 carries labels 0 and 1 → the
    # whole code goes; code 7 (twice, same label) and 9 stay, in order.  n_dropped counts
    # DISTINCT codes, not samples.
    codes = np.array([5, 5, 7, 7, 9], dtype=np.int64)
    labels = np.array([0, 1, 2, 2, 1], dtype=np.int64)
    kept_codes, kept_labels, n_dropped = iris_data.drop_ambiguous_codes(codes, labels)
    np.testing.assert_array_equal(kept_codes, [7, 7, 9])
    np.testing.assert_array_equal(kept_labels, [2, 2, 1])
    assert n_dropped == 1


def test_drop_policy_filters_both_splits_and_reaches_ceiling_one() -> None:
    # [Rosenhahn e-mail 2026-08-21] with the drop policy no code carries two classes, so the
    # encoding ceiling is 1.0 by construction and every remaining duplicate is same-class.
    split = iris_data.load_split(seed=0, ambiguous="drop")
    assert split.ambiguous == "drop"
    assert split.train_collisions == 0
    assert split.test_collisions == 0
    assert split.n_dropped_train > 0 and split.n_dropped_test > 0
    assert iris_data.encoding_accuracy_ceiling(split.train_codes, split.train_labels) == 1.0
    assert iris_data.encoding_accuracy_ceiling(split.test_codes, split.test_labels) == 1.0
    s = split.summary()
    assert s["ambiguous"] == "drop"
    assert s["n_dropped_train"] == split.n_dropped_train
    assert s["n_dropped_test"] == split.n_dropped_test


def test_default_split_is_bit_for_bit_the_merge_split() -> None:
    # The drop policy must not move any existing number: the default load_split equals an
    # explicit ambiguous="merge" call on every array, and records zero drops.
    default = iris_data.load_split(seed=0)
    merge = iris_data.load_split(seed=0, ambiguous="merge")
    assert default.ambiguous == "merge"
    for a, b in (
        (default.train_index, merge.train_index),
        (default.test_index, merge.test_index),
        (default.train_codes, merge.train_codes),
        (default.train_labels, merge.train_labels),
        (default.test_codes, merge.test_codes),
        (default.test_labels, merge.test_labels),
        (default.centers, merge.centers),
    ):
        np.testing.assert_array_equal(a, b)
    assert default.n_dropped_train == 0 and default.n_dropped_test == 0
    assert merge.n_dropped_train == 0 and merge.n_dropped_test == 0


def test_superposition_state_is_normalised_and_supported_on_the_codes() -> None:
    net = Network(n_inputs=12, n_hidden=2)
    codes = np.array([0b100100100100, 0b010010010010, 0b100100100100], dtype=np.int64)
    psi = iris_model.dataset_superposition(codes, net.n_qubits, net.input_qubits)
    assert psi.shape == (2**net.n_qubits,)
    assert np.vdot(psi, psi).real == pytest.approx(1.0)
    support = np.nonzero(np.abs(psi) > 0)[0]
    assert len(support) == 2  # distinct codes only
    n_tail = net.n_qubits - net.n_inputs
    np.testing.assert_array_equal(np.sort(support >> n_tail), [0b010010010010, 0b100100100100])
    np.testing.assert_array_equal(support & (2**n_tail - 1), 0)  # hidden/output in |0⟩


def test_target_has_three_levels_in_ratio_one_four_nine() -> None:
    # [PTQNN Fig 8, top right] measured bar heights: 15 bins at 3 levels with counts (3, 5, 7);
    # the levels are 1/86, 4/86, 9/86 (0.0116, 0.0465, 0.1047) — i.e. p(x) ∝ (label+1)²,
    # "the amplitude encodes the class label" with amplitude ∝ label+1.
    codes = np.array([1, 2, 4, 8], dtype=np.int64)
    labels = np.array([0, 1, 2, 2], dtype=np.int64)
    p = iris_model.target_distribution(codes, labels)
    assert p.shape == (2**12,)
    assert p.sum() == pytest.approx(1.0)
    assert np.count_nonzero(p) == 4
    s = 1 + 4 + 9 + 9
    np.testing.assert_allclose(p[[1, 2, 4, 8]], np.array([1, 4, 9, 9]) / s)
    assert len(set(np.round(p[p > 0] / p[p > 0].min(), 6))) == 3


def test_kl_of_the_target_against_itself_vanishes() -> None:
    codes = np.array([1, 2, 4, 8], dtype=np.int64)
    labels = np.array([0, 1, 2, 2], dtype=np.int64)
    p = iris_model.target_distribution(codes, labels)
    assert losses.kl_divergence(p, p) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("hidden", ["prototype", "classic"])
def test_fast_evaluator_agrees_with_the_reference_backend(hidden: str) -> None:
    # The fast path drops the output qubit analytically (all output rotations commute); it must
    # reproduce the statevector oracle exactly.
    net = Network(n_inputs=4, n_hidden=2, hidden=hidden)  # type: ignore[arg-type]
    rng = np.random.default_rng(3)
    codes = np.array([0b0011, 0b1010, 0b0101, 0b1111], dtype=np.int64)
    psi = iris_model.dataset_superposition(codes, net.n_qubits, net.input_qubits)
    params = net.random_params(rng)
    fast = iris_model.model_distribution(net, params, psi)
    slow = iris_model.model_distribution(net, params, psi, backend=ref.StatevectorBackend())
    np.testing.assert_allclose(fast, slow, atol=1e-12)
    assert fast.sum() == pytest.approx(1.0)
    fast_r = iris_model.model_distribution(net, params, psi, support=codes)
    slow_r = iris_model.model_distribution(net, params, psi, ref.StatevectorBackend(), codes)
    np.testing.assert_allclose(fast_r, slow_r, atol=1e-12)


def test_prototype_layer_leaks_the_input_register_and_support_post_selects_it() -> None:
    # The prototype sandwich RX(φ)/CRX/RX(−φ) does not restore the input register (the CRX
    # entangles it), so at a wide initialisation most of the probability lands outside the code
    # space.  `support=` post-selects it back — the variant we do *not* train with, because
    # nothing then penalises the leak (see the ptqnn.tasks.iris.model docstring).
    split = iris_data.load_split(seed=0)
    net = Network(n_inputs=12, n_hidden=2, hidden="prototype")
    psi = iris_model.dataset_superposition(split.train_codes, net.n_qubits, net.input_qubits)
    codes = np.unique(split.train_codes)
    params = net.random_params(np.random.default_rng(0))
    q = iris_model.model_distribution(net, params, psi)
    qr = iris_model.model_distribution(net, params, psi, support=codes)
    assert q[codes].sum() < 0.5
    assert qr[codes].sum() == pytest.approx(1.0)
    assert np.count_nonzero(qr) <= codes.size
    # a classic layer never touches the inputs, so the two variants coincide exactly
    netc = Network(n_inputs=12, n_hidden=2, hidden="classic")
    pc = netc.random_params(np.random.default_rng(0))
    psic = iris_model.dataset_superposition(split.train_codes, netc.n_qubits, netc.input_qubits)
    np.testing.assert_allclose(
        iris_model.model_distribution(netc, pc, psic),
        iris_model.model_distribution(netc, pc, psic, support=codes),
        atol=1e-12,
    )


@pytest.mark.parametrize(("hidden", "start"), [("classic", 0.24), ("prototype", 0.27)])
def test_initial_kl_matches_the_fig7_starting_values(hidden: str, start: float) -> None:
    # [PTQNN Fig 7] the training loss starts at ≈0.24 (left, classic) and ≈0.27 (right,
    # prototype); read off the figures at ±0.005.  A random initialisation of our nets must
    # land in the same decade — this is the sharpest available check of the normalisation.
    split = iris_data.load_split(seed=0)
    cfg = iris_train.IrisConfig()
    net = Network(n_inputs=12, n_hidden=4, hidden=hidden)  # type: ignore[arg-type]
    loss = iris_model.kl_loss(net, split.train_codes, split.train_labels)
    kls = [
        loss(net.random_params(np.random.default_rng(s), scale=cfg.init_scale)) for s in range(12)
    ]
    assert 0.5 * start < float(np.median(kls)) < 2.0 * start


def test_predict_labels_inverts_the_target_levels() -> None:
    codes = np.array([1, 2, 4, 8, 16, 32], dtype=np.int64)
    labels = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    values = iris_model.target_distribution(codes, labels)[codes]
    np.testing.assert_array_equal(iris_model.labels_from_values(values), labels)


def test_level_scale_fit_survives_noisy_regressed_values() -> None:
    # Regression guard: a scale fitted by plain least squares collapses (shrinking s shrinks
    # every residual, so "everything is class 0" wins).  The regressed values are a probability
    # distribution over the codes, so the fit must use Σ_x s·v(x) = Σ_x (label+1)².
    rng = np.random.default_rng(0)
    codes = np.arange(1, 25, dtype=np.int64)
    labels = np.array([0, 1, 2] * 8, dtype=np.int64)
    exact = iris_model.target_distribution(codes, labels)[codes]
    s_true = float(((labels + 1.0) ** 2).sum())
    for _ in range(10):
        noisy = exact * rng.uniform(0.75, 1.25, size=exact.size)  # ±25 %: levels separable
        noisy /= noisy.sum()
        np.testing.assert_array_equal(iris_model.labels_from_values(noisy), labels)
        assert iris_model.fit_level_scale(noisy) == pytest.approx(s_true, rel=0.15)


def test_training_smoke_kl_decreases() -> None:
    split = iris_data.load_split(seed=0)
    net = Network(n_inputs=12, n_hidden=2, hidden="prototype")
    cfg = iris_train.IrisConfig(iterations=2, n_hidden=2)
    res = iris_train.train(
        net,
        split.train_codes[:20],
        split.train_labels[:20],
        split.test_codes[:20],
        split.test_labels[:20],
        cfg,
        seed=1,
    )
    assert len(res.train_loss) == cfg.iterations + 1
    assert len(res.test_loss) == cfg.iterations + 1
    assert res.train_loss[-1] < res.train_loss[0]


def test_hidden_layer_weights_are_twelve_dimensional() -> None:
    net = Network(n_inputs=12, n_hidden=4, hidden="prototype")
    params = net.random_params(np.random.default_rng(0))
    w = iris_train.hidden_layer_weights_for_tsne(net, params)
    assert w.shape == (4, 12)
    p = iris_train.hidden_layer_weights_for_tsne(net, params, as_pattern=True)
    assert p.shape == (4, 12) and p.min() >= 0.0 and p.max() <= 1.0


def test_run_fig7_trains_both_models_with_the_same_parameter_count() -> None:
    # [PTQNN §III.C] "the prototype model (with the same amount of parameters) on the right".
    split = iris_data.load_split(seed=0)
    cfg = iris_train.IrisConfig(iterations=1, n_hidden=1)
    out = iris_train.run_fig7(
        split.train_codes[:12],
        split.train_labels[:12],
        split.test_codes[:12],
        split.test_labels[:12],
        cfg,
        seed=0,
    )
    assert set(out) == {"classic", "prototype"}
    n = {k: iris_train.network(cfg, k).n_params for k in out}
    assert n["classic"] == n["prototype"] == cfg.n_hidden * 13 + cfg.n_hidden + 1
    for res in out.values():
        assert len(res.train_loss) == len(res.test_loss) == 2


def test_tsne_embedding_places_data_and_weights_in_one_plane() -> None:
    split = iris_data.load_split(seed=0)
    net = Network(n_inputs=12, n_hidden=4, hidden="prototype")
    w = iris_train.hidden_layer_weights_for_tsne(
        net, net.random_params(np.random.default_rng(0)), as_pattern=True
    )
    data_xy, w_xy = iris_train.tsne_embedding(split.train_codes, w, seed=0)
    assert data_xy.shape == (np.unique(split.train_codes).size, 2)
    assert w_xy.shape == (4, 2)


@pytest.mark.slow
def test_prototype_network_reaches_ninety_percent_test_accuracy() -> None:
    # [PTQNN §III.C] "when applying the regressed values to the nearest class labels, the
    # performance of our model is in the range of 98% accuracy."  We demand ≥ 90 %.
    split = iris_data.load_split(seed=0)
    net = Network(n_inputs=12, n_hidden=4, hidden="prototype")
    cfg = iris_train.IrisConfig()
    res = iris_train.train(
        net,
        split.train_codes,
        split.train_labels,
        split.test_codes,
        split.test_labels,
        cfg,
        seed=0,
    )
    acc = iris_model.accuracy(net, res.params, split.test_codes, split.test_labels)
    assert acc >= 0.90, f"test accuracy {acc:.3f} (paper: ≈0.98)"
    # the read-out must also work sample by sample, not only on the batch it was trained on
    uniq, act = iris_model.basis_activations(net, res.params, split.test_codes)
    per_sample = iris_model.labels_from_values(act)[np.searchsorted(uniq, split.test_codes)]
    assert float(np.mean(per_sample == split.test_labels)) >= 0.90


# --- quantitative prototype-placement metric (pi-04; replaces Fig 9 t-SNE as evidence) ---


@pytest.mark.parametrize("hidden", ["prototype", "classic"])
def test_decode_hidden_patterns_is_one_hot_per_block_and_monotone_invariant(
    hidden: str,
) -> None:
    # The decoded pattern is a valid Iris code: exactly one 1-bit per 3-bit feature block
    # ([SNN §IV.B] one-hot-per-feature encoding).  [Rosenhahn e-mail 2026-08-21] calls the
    # cos²(φ/2) map (prototype) the "inverse" encoding; it is the Fig 9 pattern map.
    net = Network(n_inputs=12, n_hidden=4, hidden=hidden)  # type: ignore[arg-type]
    params = net.random_params(np.random.default_rng(1), scale=0.7)
    for enc in ("inverse", "direct"):
        pats = iris_train.decode_hidden_patterns(net, params, encoding=enc)  # type: ignore[arg-type]
        assert pats.shape == (4, 12) and pats.dtype == np.uint8
        np.testing.assert_array_equal(pats.reshape(4, 4, 3).sum(axis=2), 1)
    # "inverse" decoding is exactly the Fig 9 pattern map, block-argmaxed
    w = iris_train.hidden_layer_weights_for_tsne(net, params, as_pattern=True)
    np.testing.assert_array_equal(
        iris_train.decode_hidden_patterns(net, params, encoding="inverse"),
        iris_train.block_argmax(w),
    )
    # the block argmax only reads the per-block ordering, so it is invariant under any
    # monotone map of the pattern matrix (angles are NOT scale-free — 2·params is a
    # different neuron — hence the invariance is tested on m, not on the parameters)
    np.testing.assert_array_equal(iris_train.block_argmax(w), iris_train.block_argmax(w**3))


def test_covering_cost_vanishes_on_the_codes_themselves(split: iris_data.IrisSplit) -> None:
    uniq = np.unique(split.train_codes)
    cost = iris_train.covering_cost(iris_data.code_bits(uniq), split.train_codes)
    assert cost == pytest.approx(0.0)


def test_covering_cost_one_pattern_two_wrong_bits_is_one_block() -> None:
    # Halved Hamming distance: one wrong 3-bit block costs 2 bits, so a single pattern at
    # Hamming distance 2 from a single code scores d_H/2 = 1.0.
    code = int("100100100100", 2)
    pattern = np.array([[0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0]], dtype=np.uint8)  # d_H = 2
    assert iris_train.covering_cost(pattern, np.array([code], dtype=np.int64)) == pytest.approx(1.0)


def test_covering_floor_and_null_bracket_the_trained_situation(
    split: iris_data.IrisSplit,
) -> None:
    uniq = np.unique(split.train_codes)
    m = int(uniq.size)
    assert iris_train.covering_floor(split.train_codes, m) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="k > M"):
        iris_train.covering_floor(split.train_codes, m + 1)
    # n_draws ≥ 1000 (statistical-tests rule); random patterns must cover worse than the
    # best k = 4 medoids and with non-zero spread.
    floor = iris_train.covering_floor(split.train_codes, 4)
    mean, std = iris_train.covering_null(split.train_codes, 4, n_draws=1000, seed=0)
    assert mean > floor
    assert std > 0.0


@pytest.mark.parametrize("hidden", ["prototype", "classic"])
def test_prototype_placement_on_random_params_is_finite_and_bounded(hidden: str) -> None:
    split = iris_data.load_split(seed=0)
    net = Network(n_inputs=12, n_hidden=4, hidden=hidden)  # type: ignore[arg-type]
    params = net.random_params(np.random.default_rng(0), scale=0.7)
    score = iris_train.prototype_placement(net, params, split.train_codes, n_draws=1000, seed=0)
    assert score.encoding == "inverse"
    assert score.n_dead in range(5)
    assert score.n_patterns == 4 - score.n_dead
    for v in (score.R, score.R_null, score.R_null_std, score.R_floor, score.rho):
        assert np.isfinite(v)
    assert score.rho <= 1.0 + 1e-9
    # without dropping, all four patterns are scored regardless of deadness
    full = iris_train.prototype_placement(
        net, params, split.train_codes, n_draws=200, seed=0, drop_dead=False
    )
    assert full.n_patterns == 4
    assert full.n_dead == score.n_dead


# --- encoding-free firing score (session 5): what the hidden neurons actually respond to ---


@pytest.mark.parametrize("hidden", ["prototype", "classic"])
def test_firing_matrix_is_a_probability_and_matches_the_reference_backend(hidden: str) -> None:
    # P(hidden_j = 1 | code) read off the real circuit.  The fast route regroups the hidden
    # layer input qubit by input qubit and traces each input qubit out (exact: blocks on
    # different input qubits commute, see `_hidden_layer_factors`); the oracle route runs the
    # FULL 17-qubit network through the numpy reference backend.  They must agree to machine
    # precision — a marginal-only check would not catch a wrong regrouping, so the comparison
    # is run for every neuron and every code, not on an aggregate.
    split = iris_data.load_split(seed=0, ambiguous="drop")
    codes = np.unique(split.train_codes)[:3]
    net = Network(n_inputs=12, n_hidden=4, hidden=hidden)  # type: ignore[arg-type]
    params = net.random_params(np.random.default_rng(2), scale=0.7)
    fast = iris_train.firing_matrix(net, params, codes)
    assert fast.shape == (4, codes.size)
    assert fast.min() >= 0.0 and fast.max() <= 1.0
    oracle = iris_train.firing_matrix(net, params, codes, backend=ref.StatevectorBackend())
    np.testing.assert_allclose(fast, oracle, atol=1e-12)
    # distinct codes only, ordered like np.unique — duplicates do not add columns
    dup = np.concatenate([codes, codes])
    np.testing.assert_allclose(iris_train.firing_matrix(net, params, dup), fast, atol=1e-15)


def test_prototype_with_zero_prototype_angles_fires_on_the_hamming_weight_only() -> None:
    # [PTQNN Fig 2] with φ = 0 the RX(φ)/RX(−φ) sandwich is the identity, so a basis input |c⟩
    # simply triggers one CRX(θ) per 1-bit and all of them are X-rotations of the same hidden
    # qubit: the hidden state is RX(θ·|c|)|0⟩ and P(fire | c) = sin²(θ·|c|/2).  Every valid Iris
    # code has exactly four 1-bits [SNN §IV.B], so the neuron CANNOT tell the codes apart —
    # P(fire) = sin²(2θ) for all of them.  This is the degenerate case the prototype angles
    # exist to break, and it pins the firing matrix to a closed form.
    net = Network(n_inputs=12, n_hidden=1, hidden="prototype")
    codes = np.unique(iris_data.load_split(seed=0, ambiguous="drop").train_codes)
    for theta in (np.pi, np.pi / 3, 0.7):
        params = net.pack([(np.zeros(12), theta)], (np.array([0.3]), 0.1))
        fire = iris_train.firing_matrix(net, params, codes)
        assert fire.shape == (1, codes.size)
        np.testing.assert_allclose(fire, np.sin(2.0 * theta) ** 2, atol=1e-12)
        # a constant response covers as loudly as it fires and discriminates nothing
        coverage, selectivity = iris_train.coverage_selectivity(fire)
        assert coverage == pytest.approx(np.sin(2.0 * theta) ** 2, abs=1e-12)
        assert selectivity == pytest.approx(0.0, abs=1e-12)


def test_relative_dead_tolerance_never_flags_the_strongest_neuron() -> None:
    # The trained prototype layer of experiments/runs/iris/20260918T085641Z-drop.  The Iris
    # target is amplitude-encoded [PTQNN Fig 8], so every coupling is small: drive_j =
    # |sin(θ_j/2)·sin(w_j/2)| ≤ 0.044.  The former absolute tol = 0.05 called all four dead;
    # the relative tolerance keeps the strongest and flags only the one 16× below it.
    net = Network(n_inputs=12, n_hidden=4, hidden="prototype")
    theta = [0.45, -0.35, 0.31, -0.39]
    w_out = np.array([0.16, -0.03, -0.56, -0.31])
    params = net.pack([(np.zeros(12), t) for t in theta], (w_out, 0.0))
    drive = iris_train.output_drive(net, params)
    np.testing.assert_allclose(drive, np.abs(np.sin(np.array(theta) / 2) * np.sin(w_out / 2)))
    assert drive.max() < 0.05  # would have been "all dead" under the old absolute tolerance
    dead = iris_train.dead_hidden_neurons(net, params)
    assert not bool(dead[int(drive.argmax())]), "the strongest neuron is never dead"
    np.testing.assert_array_equal(dead, drive < 0.1 * drive.max())
    assert dead.sum() == 1
    # the absolute floor is still available and reproduces the old verdict
    assert iris_train.dead_hidden_neurons(net, params, tol_abs=0.05).all()
    # a strong neuron cannot be flagged by the relative rule however small the layer's scale
    for scale in (1e-3, 1.0, 10.0):
        p = net.pack([(np.zeros(12), scale * t) for t in theta], (scale * w_out, 0.0))
        d = iris_train.dead_hidden_neurons(net, p)
        assert not bool(d[int(iris_train.output_drive(net, p).argmax())])


def test_prototype_placement_reports_all_dead_instead_of_raising(
    split: iris_data.IrisSplit,
) -> None:
    # An all-dead layer used to raise and killed the run AFTER a 40-minute training
    # (experiments/runs/iris/20260918T085641Z-drop lost its results.json that way).
    net = Network(n_inputs=12, n_hidden=4, hidden="prototype")
    params = net.random_params(np.random.default_rng(0), scale=0.7)
    score = iris_train.prototype_placement(
        net, params, split.train_codes, n_draws=200, seed=0, tol_abs=1.0
    )
    assert score.all_dead is True
    assert score.n_dead == 4 and score.n_patterns == 4
    assert np.isfinite(score.R) and np.isfinite(score.rho)
    # the ordinary case still reports all_dead=False
    alive = iris_train.prototype_placement(net, params, split.train_codes, n_draws=200, seed=0)
    assert alive.all_dead is False


@pytest.mark.parametrize("hidden", ["prototype", "classic"])
def test_firing_placement_z_scores_the_trained_layer_against_a_random_null(hidden: str) -> None:
    split = iris_data.load_split(seed=0, ambiguous="drop")
    net = Network(n_inputs=12, n_hidden=4, hidden=hidden)  # type: ignore[arg-type]
    params = net.random_params(np.random.default_rng(7), scale=0.7)
    score = iris_train.firing_placement(net, params, split.train_codes, n_draws=50, seed=0)
    assert 0.0 <= score.coverage <= 1.0 and 0.0 <= score.selectivity <= 1.0
    assert score.n_draws == 50 and score.n_hidden == 4
    assert score.n_codes == int(np.unique(split.train_codes).size)
    assert score.coverage_null_std > 0.0 and score.selectivity_null_std > 0.0
    # the z-scores are exactly the standardised summaries, not a re-derivation
    assert score.coverage == pytest.approx(
        score.coverage_null + score.rho_f * score.coverage_null_std
    )
    assert score.selectivity == pytest.approx(
        score.selectivity_null + score.rho_f_selectivity * score.selectivity_null_std
    )
    # the null is the initialisation itself, so a draw from it scores ≈ 0 rather than ≫ 0
    drawn = net.random_params(np.random.default_rng(0), scale=0.7)
    typical = iris_train.firing_placement(net, drawn, split.train_codes, n_draws=50, seed=0)
    assert abs(typical.rho_f) < 5.0
    with pytest.raises(ValueError, match="n_draws"):
        iris_train.firing_placement(net, params, split.train_codes, n_draws=0)


def test_rescore_writes_results_without_retraining(tmp_path: Path) -> None:
    # --rescore must reach results-rescore.json from params-*.npy alone: the run that crashed
    # after training had neither results.json nor meta.json.
    script = Path(__file__).resolve().parents[1] / "experiments" / "iris_fig7_fig8_fig9.py"
    spec = importlib.util.spec_from_file_location("iris_fig7_fig8_fig9", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["iris_fig7_fig8_fig9"] = module
    spec.loader.exec_module(module)
    rng = np.random.default_rng(0)
    for kind in ("classic", "prototype"):
        net = Network(n_inputs=12, n_hidden=4, hidden=kind)  # type: ignore[arg-type]
        np.save(tmp_path / f"params-{kind}.npy", net.random_params(rng, scale=0.7))
    rc = module.main(
        [
            "--rescore",
            str(tmp_path),
            "--seed",
            "0",
            "--ambiguous",
            "drop",
            "--placement-draws",
            "50",
            "--firing-draws",
            "3",
        ]
    )
    assert rc == 0
    results = json.loads((tmp_path / "results-rescore.json").read_text())
    meta = json.loads((tmp_path / "meta-rescore.json").read_text())
    assert set(results) == {"accuracy", "placement", "firing"}
    assert set(results["placement"]) == {
        f"{k}-{e}" for k in ("classic", "prototype") for e in ("inverse", "direct")
    }
    assert set(results["firing"]) == {"classic", "prototype"}
    for kind in ("classic", "prototype"):
        assert 0.0 <= results["accuracy"][kind]["test"] <= 1.0
        assert "error" not in results["firing"][kind]
        assert results["firing"][kind]["n_draws"] == 3
    assert meta["flags"]["ambiguous"] == "drop" and meta["n_hidden_from_params"] == 4
    assert meta["git_hash"] and meta["split"]["ambiguous"] == "drop"
    # nothing was retrained: the parameter files are untouched and no new params appeared
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "meta-rescore.json",
        "params-classic.npy",
        "params-prototype.npy",
        "results-rescore.json",
    ]
