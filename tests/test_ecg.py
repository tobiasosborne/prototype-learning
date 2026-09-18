"""ECG denoising ``[PTQNN §III.D, Fig 12, 13, Table I, Eq 6–8]``."""

from __future__ import annotations

import numpy as np
import pytest

from ptqnn.backends import reference as ref
from ptqnn.circuits.neurons import Network
from ptqnn.tasks.ecg import data, model, train

# --- data -------------------------------------------------------------------------------------

# The co-author's "original" dataset [Rosenhahn e-mail 2026-08-21]; not in git (see
# data/README.md), so every test that reads it is skipped when it has not been staged.
DATA_DIR = data.MAILED_DIR
needs_mailed = pytest.mark.skipif(
    not DATA_DIR.exists(), reason="mailed ECG dataset not staged in data/raw/ecg-rosenhahn/"
)


def test_synthetic_ecg_is_in_the_unit_interval() -> None:
    s = data.synthetic_ecg(1800, seed=0)
    assert s.shape == (1800,)
    assert s.min() >= 0.0 and s.max() <= 1.0
    # Fig 13's y-axis: the trace lives between ~0.13 and ~0.93, never touching the frame.
    assert 0.05 < s.min() < 0.2
    assert 0.85 < s.max() < 0.98


def test_synthetic_ecg_is_smooth() -> None:
    """Clean signal: no sample-to-sample jumps anywhere near the noise amplitude of Fig 13."""
    s = data.synthetic_ecg(1800, seed=0)
    assert np.abs(np.diff(s)).max() < 0.02


def test_synthetic_ecg_reproduces_the_fig13_landmarks() -> None:
    """Two R peaks at t ≈ 380 and t ≈ 1270, both ≈ 0.9 (read off Fig 13, ±20 samples / ±0.03)."""
    s = data.synthetic_ecg(1800, seed=0, rr_jitter=0.0, amplitude_jitter=0.0)
    peaks = [t for t in range(1, 1799) if s[t] > 0.7 and s[t] >= s[t - 1] and s[t] > s[t + 1]]
    assert len(peaks) == 2
    assert abs(peaks[0] - 380) <= 20
    assert abs(peaks[1] - 1270) <= 20
    assert all(abs(s[t] - 0.9) < 0.05 for t in peaks)


def test_synthetic_ecg_is_seed_dependent_but_deterministic() -> None:
    a = data.synthetic_ecg(600, seed=1)
    b = data.synthetic_ecg(600, seed=1)
    c = data.synthetic_ecg(600, seed=2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_add_noise_is_additive_gaussian_with_the_requested_sigma() -> None:
    s = data.synthetic_ecg(4000, seed=0)
    n = data.add_noise(s, sigma=0.02, seed=3)
    assert n.shape == s.shape
    assert abs(float(np.std(n - s)) - 0.02) < 0.002
    assert abs(float(np.mean(n - s))) < 0.002
    # the default matches the ±0.05 visual envelope of Fig 13 (≈ 2.5 σ)
    assert 2.0 < 0.05 / data.DEFAULT_SIGMA < 3.5


def test_windows_shape_and_alignment() -> None:
    s = data.synthetic_ecg(200, seed=0)
    X, y = data.windows(s, half_width=2)
    assert X.shape == (200, 5)
    assert y.shape == (200,)
    # y[i] is the centre of window i
    assert np.allclose(y, s)
    assert np.allclose(X[:, 2], s)
    assert np.allclose(X[10], s[8:13])


def test_windows_valid_mode_drops_the_edges() -> None:
    s = np.arange(10, dtype=np.float64) / 10
    X, y = data.windows(s, half_width=2, edge="valid")
    assert X.shape == (6, 5)
    assert np.allclose(y, s[2:8])
    assert np.allclose(X[0], s[0:5])


def test_windows_reflect_padding_at_the_boundary() -> None:
    s = np.arange(10, dtype=np.float64)
    X, _ = data.windows(s, half_width=2, edge="reflect")
    assert np.allclose(X[0], [2, 1, 0, 1, 2])
    assert np.allclose(X[-1], [7, 8, 9, 8, 7])


@needs_mailed
def test_load_mailed_returns_the_full_and_the_default_cut() -> None:
    """Both traces are (1, 7700) float64 row vectors [Rosenhahn e-mail 2026-08-21]."""
    clean, noisy = data.load_mailed(n_samples=None)
    assert clean.shape == (7700,) and noisy.shape == (7700,)
    assert clean.dtype == np.float64 and noisy.dtype == np.float64
    clean, noisy = data.load_mailed()
    assert clean.shape == (1800,) and noisy.shape == (1800,)


@needs_mailed
def test_load_mailed_first_samples_are_identical_in_both_traces() -> None:
    """[Rosenhahn e-mail 2026-08-21]: samples 1-3 of *both* traces are 0.13049, 0.16907, 0.12890."""
    clean, noisy = data.load_mailed()
    first = np.array([0.13049, 0.16907, 0.12890])
    assert np.allclose(clean[:3], first, atol=1e-5)
    assert np.allclose(noisy[:3], first, atol=1e-5)


@needs_mailed
def test_load_mailed_ranges_and_measured_noise_level() -> None:
    """The clean trace is already in [0, 1]; the noisy one is clipped to exactly [0, 1] and
    noisy - filtered has std ≈ 0.0354 [Rosenhahn e-mail 2026-08-21].  No renormalisation."""
    clean, noisy = data.load_mailed(n_samples=None)
    assert clean.min() >= 0.0 and clean.max() <= 1.0
    assert noisy.min() == 0.0 and noisy.max() == 1.0
    assert 0.030 < float(np.std(noisy - clean)) < 0.040


@needs_mailed
def test_prepare_mailed_uses_the_file_noise_instead_of_add_noise() -> None:
    """The mailed files already contain the noise, so ``sigma`` is *measured*, not applied."""
    ds = train.prepare(source="mailed", n_samples=1800)
    assert ds.source == "mailed"
    # same window count as the synthetic default (1800 samples, reflect padding)
    assert ds.X.shape[0] == train.prepare(source="synthetic", n_samples=1800).X.shape[0]
    # sigma is exactly the measured std, and matches the full-trace value to 0.005
    assert abs(ds.sigma - float(np.std(ds.noisy - ds.clean))) < 1e-12
    assert abs(ds.sigma - 0.03545) < 0.005
    assert ds.sigma != data.DEFAULT_SIGMA


@needs_mailed
def test_mailed_provenance_pins_the_files_and_the_sender() -> None:
    prov = data.mailed_provenance()
    assert prov["ecgfilteredData.mat"] == (
        "9ed65523b30e588012dba0ece34e8d0bb6951254357cf0757ef1692b7439b456"
    )
    assert prov["ecgnoiseData.mat"] == (
        "cc94548d1c40b63b597b532edbf0bd198ac3a12ac9253d0efc4b40255d0753b9"
    )
    assert prov["sender"] == "B. Rosenhahn, e-mail 2026-08-21"


def test_contiguous_split_has_no_window_overlap() -> None:
    tr, te = data.contiguous_split(100, train_frac=0.5, half_width=2)
    assert tr.max() + 2 < te.min() - 2
    assert set(tr).isdisjoint(te)
    assert len(tr) + len(te) <= 100


# --- metrics ----------------------------------------------------------------------------------


def test_psnr_of_identical_signals_is_infinite() -> None:
    s = data.synthetic_ecg(300, seed=0)
    assert model.psnr(s, s) == np.inf


def test_psnr_matches_the_closed_form_of_eq_6() -> None:
    rng = np.random.default_rng(0)
    a = rng.random(500)
    b = a + 0.01
    r = float(a.max() - a.min())
    assert abs(model.psnr(a, b) - 10 * np.log10(r**2 / 1e-4)) < 1e-9
    # Eq 6 and Eq 7 are the same number
    assert abs(model.psnr(a, b) - 20 * np.log10(r / 0.01)) < 1e-9


def test_ssim_of_identical_signals_is_one_and_is_symmetric() -> None:
    s = data.synthetic_ecg(400, seed=0)
    t = data.add_noise(s, sigma=0.02, seed=1)
    assert abs(model.ssim(s, s) - 1.0) < 1e-12
    assert abs(model.ssim(s, t) - model.ssim(t, s)) < 1e-12
    assert 0.0 < model.ssim(s, t) < 1.0
    assert abs(model.ssim_windowed(s, s) - 1.0) < 1e-12
    assert model.ssim_windowed(s, t) < model.ssim(s, t)


# --- encoding ---------------------------------------------------------------------------------


def test_encode_defaults_to_the_amplitude_convention_of_snn_fig4() -> None:
    """``[SNN Fig 4, §III.A]`` prints ``CRX(1.369)`` for the value 0.4 and ``CRX(1.7722)`` for
    0.6, i.e. ``θ = 2 asin(√s)`` and ``P(|1⟩) = sin²(θ/2) = s``."""
    ops = model.encode([0.4, 0.6, 0.1, 0.2, 0.3])
    assert [op.name for op in ops] == ["rx"] * 5
    assert [op.qubits for op in ops] == [(0,), (1,), (2,), (3,), (4,)]
    assert abs(ops[0].params[0] - 1.3694) < 5e-5
    assert abs(ops[1].params[0] - 1.7722) < 5e-5
    w = np.array([0.4, 0.6, 0.1, 0.2, 0.3])
    assert np.allclose(model.decode(model.angles(w)), w)


def test_linear_encoding_is_available_as_an_ablation() -> None:
    """θ = π·s, the reading ``[PTQNN Fig 12]`` was thought to support (PLAN.md §0)."""
    w = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    ops = model.encode(w, encoding="linear")
    assert np.allclose([op.params[0] for op in ops], np.pi * w)
    assert np.allclose(model.decode(np.pi * w, encoding="linear"), w)


def test_amplitude_encoding_clips_out_of_range_noisy_samples() -> None:
    assert np.allclose(model.angles([-0.1, 1.4], encoding="amplitude"), [0.0, np.pi])


def test_fig12_angles_cannot_decide_the_encoding() -> None:
    """``[PTQNN Fig 12]`` prints RX(0.54454, 0.78147, 0.51643, 0.70226, 0.78199) while the window
    it highlights reads {0.2108, 0.3344, 0.3329, 0.3631, 0.2597}
    (docs/physics/rosenhahn_hirche_ptqnn.md §6.2/§6.4): the angle order is not the value order,
    so no monotone encoding fits and Fig 12 cannot overrule ``[SNN Fig 4]``.  This test pins the
    contradiction so nobody re-derives an encoding from Fig 12 again."""
    printed = np.array([0.54454, 0.78147, 0.51643, 0.70226, 0.78199])
    window = np.array([0.2108, 0.3344, 0.3329, 0.3631, 0.2597])
    assert not np.array_equal(np.argsort(printed), np.argsort(window))


# --- forward pass -----------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["prototype", "classic"])
def test_batched_prediction_agrees_with_the_reference_backend(kind: str) -> None:
    net = Network(n_inputs=5, n_hidden=2, hidden=kind)  # type: ignore[arg-type]
    rng = np.random.default_rng(0)
    params = net.random_params(rng)
    X = rng.random((7, 5))
    for encoding in ("amplitude", "linear"):
        fast = model.predict(net, params, X, encoding=encoding)  # type: ignore[arg-type]
        slow = model.predict(
            net,
            params,
            X,
            backend=ref.StatevectorBackend(),
            encoding=encoding,  # type: ignore[arg-type]
        )
        assert np.allclose(fast, slow, atol=1e-12)
        assert np.all((fast >= 0) & (fast <= 1))


# --- training ---------------------------------------------------------------------------------


def test_one_prototype_training_beats_the_identity_denoiser() -> None:
    """Red→green smoke test: on a short, strongly noised synthetic segment the trained
    1-prototype network must reconstruct the clean signal better than the noisy input itself.

    σ = 0.06 rather than the Fig-13 default of 0.02: with a single prototype the model is
    bias-limited at MSE ≈ 1e-3, so at the default noise level "beating the identity" is a
    coin toss (and Table I's single-neuron rows, 32.72/32.9 dB, sit exactly at the input PSNR —
    the paper's own 1-neuron models barely denoise either).  Three times the noise separates a
    working denoiser from a broken one in a few seconds.
    """
    clean = data.synthetic_ecg(600, seed=0)
    noisy = data.add_noise(clean, sigma=0.06, seed=1)
    X, y = data.supervised_windows(noisy, clean)
    net = Network(n_inputs=5, n_hidden=1, hidden="prototype")
    res = train.train(net, X, y, iterations=20, seed=0, subsample=128, init_candidates=64)
    out = train.evaluate(net, res.params, X, y, clean)
    identity = float(np.mean((X[:, 2] - y) ** 2))
    assert out["mse"] < 0.6 * identity
    assert out["psnr"] > model.psnr(y, X[:, 2], np.ptp(clean))


@pytest.mark.slow
def test_table1_ordering_three_prototypes_beat_one() -> None:
    rows = train.table1(n_hidden=(1, 3), kinds=("prototype",), seed=0, iterations=25)
    by_size = {r["n_hidden"]: r for r in rows}
    assert by_size[3]["psnr"] >= by_size[1]["psnr"]
