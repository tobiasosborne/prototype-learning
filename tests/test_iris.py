"""Red-phase tests for the Iris task ``[PTQNN §III.C, Fig 7–9]``, ``[SNN §IV.B]``."""

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
    per_sample = iris_model.labels_from_values(act)[
        np.searchsorted(uniq, split.test_codes)
    ]
    assert float(np.mean(per_sample == split.test_labels)) >= 0.90
